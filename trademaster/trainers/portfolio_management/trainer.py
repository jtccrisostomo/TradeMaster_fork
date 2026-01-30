from typing import Optional
try:
    from gymnasium.wrappers import EnvCompatibility
except Exception:  # gymnasium >=1.2 may drop EnvCompatibility; fall back to identity
    def EnvCompatibility(env):
        return env
import torch
import numpy as np
import pandas as pd
from trademaster.environments.portfolio_management.environment import PortfolioManagementEnvironment
from ray.tune.registry import register_env
import ray
import os
from trademaster.utils import get_attr, plot_metric_against_baseline
from ..builder import TRAINERS
from ..custom import Trainer
import random
import shutil
from pathlib import Path
import logging
import glob

ROOT = Path(__file__).resolve().parents[3]


def env_creator(env_name):
    if env_name == "portfolio_management":
        env = PortfolioManagementEnvironment
    else:
        raise NotImplementedError
    return env


def select_algorithms(alg_name):
    alg_name = alg_name.upper()
    if alg_name == "A2C":
        from ray.rllib.algorithms.a3c import A2C as trainer
    elif alg_name == "DDPG":
        from ray.rllib.algorithms.ddpg import DDPG as trainer
    elif alg_name == "PG":
        from ray.rllib.algorithms.pg import PG as trainer
    elif alg_name == "PPO":
        from ray.rllib.algorithms.ppo import PPO as trainer
    elif alg_name == "SAC":
        from ray.rllib.algorithms.sac import SAC as trainer
    elif alg_name == "TD3":
        from ray.rllib.algorithms.td3 import TD3 as trainer
    else:
        raise NotImplementedError(f"Unknown alg: {alg_name}")
    return trainer


# no noisy logs
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

# register RLlib env (gymnasium API via wrapper)
register_env(
    "portfolio_management",
    lambda config: EnvCompatibility(env_creator("portfolio_management")(config))
)


def _resolve_ray_address(raw: Optional[str]) -> Optional[str]:
    """Normalize address but preserve client URLs as-is."""
    if not raw:
        return None
    s = str(raw).strip()
    if s.lower() in ("", "none", "local", "auto"):
        return None
    return s


@TRAINERS.register_module()
class PortfolioManagementTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementTrainer, self).__init__()

        self.device = get_attr(kwargs, "device", None)

        self.configs = get_attr(kwargs, "configs", None) or {}
        self.agent_name = get_attr(kwargs, "agent_name", "ppo")
        self.epochs = int(get_attr(kwargs, "epochs", 20))
        self.dataset = get_attr(kwargs, "dataset", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(str(ROOT), self.work_dir) if self.work_dir else os.path.join(str(ROOT), "work_dir/pm_default")
        self.seed = int(get_attr(kwargs, "seed", 12345))
        self.random_seed = self.seed
        self.if_remove = bool(get_attr(kwargs, "if_remove", False))
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))
        self.verbose = bool(get_attr(kwargs, "verbose", False))

        # --- Ray init (client -> GCS -> optional local fallback) ---
        if not ray.is_initialized():
            cfg_ray = (self.configs.get("ray_kwargs", {}) or {})
            raw_addr = cfg_ray.get("address")
            if raw_addr in (None, "None", "", "local"):
                raw_addr = os.getenv("RAY_ADDRESS") or None

            def _try_init(addr: Optional[str]) -> bool:
                try:
                    if addr:
                        print(f"DEBUG: trainer.py - ray.init(address={addr!r})")
                        ray.init(address=addr, ignore_reinit_error=True, include_dashboard=False)
                    else:
                        print("DEBUG: trainer.py - ray.init() local")
                        ray.init(ignore_reinit_error=True, include_dashboard=False)
                    print("DEBUG: trainer.py - ray.init successful!")
                    return True
                except Exception as e:
                    import traceback as _tb
                    print(f"DEBUG: trainer.py - ray.init FAILED with address={addr!r}: {e}\n{_tb.format_exc()}")
                    return False

            addr = _resolve_ray_address(raw_addr)
            ok = False

            # 1) Try given address (may be client ray:// or GCS host:port)
            if _try_init(addr):
                ok = True
            else:
                # 2) If client URL, try native GCS fallback host:<RAY_GCS_PORT or 6381>
                gcs_addr = None
                if isinstance(addr, str) and addr.startswith("ray://"):
                    try:
                        without_scheme = addr.split("://", 1)[1]
                        host = without_scheme.split(":", 1)[0]
                        gcs_port = int(os.getenv("RAY_GCS_PORT", "6381"))
                        gcs_addr = f"{host}:{gcs_port}"
                        print(f"DEBUG: trainer.py - Retrying via native GCS at {gcs_addr}")
                        ok = _try_init(gcs_addr)
                    except Exception:
                        pass

            if not ok:
                # 3) Optional local fallback (only if explicitly allowed)
                if os.getenv("RAY_ALLOW_LOCAL_FALLBACK", "0") == "1":
                    print("DEBUG: trainer.py - Falling back to local Ray (RAY_ALLOW_LOCAL_FALLBACK=1).")
                    os.environ.pop("RAY_ADDRESS", None)
                    try:
                        import ray.util.client as _client
                        if getattr(_client.ray, "is_connected", lambda: False)():
                            _client.ray.disconnect()
                    except Exception:
                        pass
                    ray.shutdown()
                    if not _try_init(None):
                        raise RuntimeError("Failed to initialize local Ray.")
                else:
                    raise RuntimeError("Could not connect to Ray (client and GCS both failed). Set RAY_ALLOW_LOCAL_FALLBACK=1 to allow local fallback.")

        # RLlib config
        self.trainer_name = select_algorithms(self.agent_name)
        self.configs["env"] = "portfolio_management"
        self.configs["env_config"] = dict(dataset=self.dataset, task="train")
        # Env checker can be overzealous; we wrap with EnvCompatibility, but also disable checks to be safe.
        self.configs["disable_env_checking"] = True
        self.configs.setdefault("framework", "torch")

        self.init_before_training()

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if self.if_remove:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.verbose:
                print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            if self.verbose:
                print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):
        valid_score_list = []
        save_dict_list = []
        checkpoint_paths = []

        # Build trainer
        self.trainer = self.trainer_name(env=self.configs["env"], config=self.configs)

        for epoch in range(1, self.epochs + 1):
            print(f"Train Episode: [{epoch}/{self.epochs}]")
            self.trainer.train()

            # validation rollout (direct env, not RLlib worker)
            config = dict(dataset=self.dataset, task="valid")
            self.valid_environment = env_creator("portfolio_management")(config)
            print(f"Valid Episode: [{epoch}/{self.epochs}]")

            state = self.valid_environment.reset()
            episode_reward_sum = 0.0
            while True:
                action = self.trainer.compute_single_action(state)
                # normalize to weights
                action = np.exp(action) / np.sum(np.exp(action))
                state, reward, done, information = self.valid_environment.step(action)
                episode_reward_sum += reward
                if done:
                    break

            # log & record
            if isinstance(information, dict) and "table" in information:
                print(information["table"])
            save_dict_list.append(information)
            valid_score_list.append(information.get("sharpe_ratio", float("-inf")))

            # checkpoint save (robust)
            pattern = os.path.join(self.checkpoints_path, "checkpoint_*")
            before = set(glob.glob(pattern))
            ret = self.trainer.save(checkpoint_dir=self.checkpoints_path)
            ckpt_dir = ret if isinstance(ret, str) else None
            if not ckpt_dir or not os.path.exists(ckpt_dir):
                after = set(glob.glob(pattern))
                new_dirs = sorted(after - before, key=os.path.getmtime)
                if new_dirs:
                    ckpt_dir = new_dirs[-1]
                else:
                    existing = sorted(after, key=os.path.getmtime)
                    ckpt_dir = existing[-1] if existing else self.checkpoints_path
            checkpoint_paths.append(ckpt_dir)

        # pick best by Sharpe
        max_index = int(np.argmax(valid_score_list))
        best_ckpt_dir = checkpoint_paths[max_index]

        # plot validation curve
        best_info = save_dict_list[max_index]
        plot_metric_against_baseline(
            total_asset=best_info.get("total_assets"),
            buy_and_hold=None,
            alg=self.agent_name.upper(),
            task="valid",
            color="darkcyan",
            save_dir=self.work_dir,
        )

        # persist pointer
        with open(os.path.join(self.checkpoints_path, "best_path.txt"), "w") as fh:
            fh.write(str(best_ckpt_dir) + "\n")

        # convenience symlink
        try:
            best_link = os.path.join(self.checkpoints_path, "best")
            if os.path.islink(best_link) or os.path.exists(best_link):
                if os.path.islink(best_link) or os.path.isfile(best_link):
                    os.remove(best_link)
                else:
                    shutil.rmtree(best_link)
            if hasattr(os, "symlink"):
                os.symlink(best_ckpt_dir, best_link)
        except Exception:
            pass

    def test(self):
        # Create trainer and restore best checkpoint
        self.trainer = self.trainer_name(env=self.configs["env"], config=self.configs)

        best_txt = os.path.join(self.checkpoints_path, "best_path.txt")
        if os.path.isfile(best_txt):
            with open(best_txt, "r") as fh:
                best_ckpt_dir = fh.read().strip()
        else:
            candidates = sorted(glob.glob(os.path.join(self.checkpoints_path, "checkpoint_*")))
            if not candidates:
                raise RuntimeError(f"No checkpoints found under {self.checkpoints_path}")
            best_ckpt_dir = candidates[-1]

        self.trainer.restore(best_ckpt_dir)

        # ---- test roll-out (your env, not RLlib) ----
        config = dict(dataset=self.dataset, task="test")
        self.test_environment = env_creator("portfolio_management")(config)
        print("Test Best Episode")

        state = self.test_environment.reset()
        episode_reward_sum = 0.0
        while True:
            action = self.trainer.compute_single_action(state)
            action = np.exp(action) / np.sum(np.exp(action))
            state, reward, done, info = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                plot_metric_against_baseline(
                    total_asset=info.get("total_assets", None),
                    buy_and_hold=None,
                    alg=self.agent_name.upper(),
                    task="test",
                    color="darkcyan",
                    save_dir=self.work_dir,
                )
                break

        # Optional: pretty table in logs if provided by env
        tbl = info.get("table") if isinstance(info, dict) else None
        if tbl is not None:
            print(tbl)

        # ---- Collect time series from env ----
        rewards = self.test_environment.save_asset_memory()
        assets = rewards.get("total assets", rewards.get("total_assets"))
        assets = np.asarray(assets, dtype=float)

        df_return = self.test_environment.save_portfolio_return_memory()
        if isinstance(df_return, pd.DataFrame) and "daily_return" in df_return.columns:
            daily = df_return["daily_return"].values.astype(float)
        else:
            # Fallback: derive daily returns from asset curve
            daily = pd.Series(assets).pct_change().dropna().values

        # Align lengths conservatively
        n = min(len(assets), len(daily))
        assets = assets[:n]
        daily = daily[:n]

        # ---- Compute metrics ----
        TRADING_DAYS = 252.0
        base = float(assets[0]) if len(assets) else np.nan
        final_value = float(assets[-1]) if len(assets) else np.nan
        cum_series = (assets / base) - 1.0 if len(assets) else np.array([], dtype=float)

        mu = float(np.mean(daily)) if len(daily) else np.nan
        sigma = float(np.std(daily, ddof=0)) if len(daily) else np.nan
        sharpe = (mu / sigma * np.sqrt(TRADING_DAYS)) if (sigma and sigma > 0) else np.nan

        if len(assets):
            roll_max = np.maximum.accumulate(assets)
            drawdowns = assets / roll_max - 1.0
            max_dd = float(np.min(drawdowns))
        else:
            max_dd = np.nan

        # ---- Write CSV with baked-in metrics ----
        df = pd.DataFrame(
            {
                "daily_return": daily,
                "total_assets": assets,
            }
        )
        # cumulative_returns per-row (w.r.t. starting capital)
        if len(cum_series):
            df["cumulative_returns"] = cum_series
        else:
            df["cumulative_returns"] = np.nan

        # Summary row (last row will contain the metrics your aggregator reads)
        summary = pd.DataFrame(
            [
                {
                    "daily_return": np.nan,
                    "total_assets": final_value,
                    "cumulative_returns": float(cum_series[-1]) if len(cum_series) else np.nan,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                }
            ]
        )
        df = pd.concat([df, summary], ignore_index=True)

        out_csv = os.path.join(self.work_dir, "test_result.csv")
        df.to_csv(out_csv, index=False)
        print("test end")
        return daily
