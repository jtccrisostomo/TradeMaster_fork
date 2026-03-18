from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict

# Make sure repo code is importable
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)

from trademaster.utils import get_attr, print_metrics
from ..custom import Environments
from ..builder import ENVIRONMENTS
from gymnasium import spaces  # gymnasium-compatible spaces


"""
Environment based on FinRL's portfolio allocation env, with fixes:
- Maintain a fixed, sorted ticker universe across all steps.
- Reindex each day's frame to that universe and fill missing tickers
  with previous day's prices so shapes always match.
- Keep the classic Gym API (reset -> obs, step -> obs, reward, done, info);
  your trainer wraps with EnvCompatibility already.
"""


@ENVIRONMENTS.register_module()
class PortfolioManagementEnvironment(Environments):
    def __init__(self, config):
        super().__init__()

        self.dataset = get_attr(config, "dataset", None)
        self.task = get_attr(config, "task", "train")
        self.day = 0

        # Paths from dataset cfg
        if self.task.startswith("train"):
            df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            df_path = get_attr(self.dataset, "valid_path", None)
        else:
            df_path = get_attr(self.dataset, "test_path", None)

        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        # Load CSV. Keep the original integer "day" index (first column).
        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(config, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
        else:
            self.df = pd.read_csv(df_path, index_col=0)
        # ---- Enforce consistent ticker universe across splits ----
        tickers = getattr(self.dataset, "ticker_universe", None)
        if tickers:
            self.df = self.df[self.df["tic"].astype(str).isin(set(map(str, tickers)))].copy()

        # It helps to keep rows ordered deterministically
        if "date" in self.df.columns and "tic" in self.df.columns:
            self.df.sort_values(["date", "tic"], inplace=True)


        # Stable universe + shapes
        self.tickers = sorted(self.df["tic"].unique().tolist())
        self.stock_dim = len(self.tickers)
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim + 1  # cash + assets

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.action_space_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.tech_indicator_list), self.state_space_shape),
            dtype=np.float32,
        )

        # Initialize per-day view and last prices
        self._cur_day_df = self._day_slice(self.day)  # reindexed to self.tickers
        self.last_prices = self._cur_day_df["close"].astype(float).to_numpy()

        self.state = self._build_state_from_day_df(self._cur_day_df)
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0.0]
        self.weights_memory = [[1.0] + [0.0] * self.stock_dim]
        self.date_memory = [self._day_date(self.day)]
        self.transaction_cost_memory = []

    # ---------- helpers ----------

    def _raw_day_df(self, day: int) -> pd.DataFrame:
        """Return the raw slice for a day (may be Series if only 1 row)."""
        sl = self.df.loc[day, :]
        if isinstance(sl, pd.Series):
            sl = sl.to_frame().T
        return sl

    def _day_slice(self, day: int) -> pd.DataFrame:
        """
        Return a DataFrame indexed by 'tic' and reindexed to the full ticker universe.
        Columns must include all tech indicators + 'close' and 'date'.
        Missing tickers (not trading this day) get NaNs to be filled later.
        """
        sl = self._raw_day_df(day)
        sl = sl.set_index("tic").reindex(self.tickers)
        return sl

    def _day_date(self, day: int):
        sl = self._raw_day_df(day)
        return sl["date"].iloc[0] if "date" in sl.columns else day

    def _build_state_from_day_df(self, day_df: pd.DataFrame) -> np.ndarray:
        # Ensure required cols exist
        need = set(self.tech_indicator_list)
        missing = [c for c in need if c not in day_df.columns]
        if missing:
            # Create missing cols filled with zeros
            for c in missing:
                day_df[c] = 0.0

        # Fill NaNs (e.g., missing tickers) with previous known values where possible,
        # otherwise zeros so obs always has a value.
        feat = day_df[self.tech_indicator_list].copy()
        feat = feat.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        # Shape: (len(features), stock_dim)
        state = np.vstack([feat[c].astype(float).to_numpy() for c in self.tech_indicator_list])
        return state

    @staticmethod
    def _softmax(actions: np.ndarray) -> np.ndarray:
        a = actions - np.max(actions)  # numerical stability
        e = np.exp(a)
        return e / (np.sum(e) + 1e-12)

    @staticmethod
    def _normalize(weights_like: list | np.ndarray) -> np.ndarray:
        w = np.asarray(weights_like, dtype=float)
        s = np.sum(w)
        return w / (s + 1e-12)

    # ---------- gym API ----------

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0.0]
        self.transaction_cost_memory = []
        self.weights_memory = [[1.0] + [0.0] * self.stock_dim]

        self.day = 0
        self._cur_day_df = self._day_slice(self.day)
        # If first day has NaNs for some tickers, set them to 0 in obs but
        # keep last_prices from available data.
        close0 = self._cur_day_df["close"].astype(float)
        # Any NaNs in first close -> fill with mean of available closes to initialize
        init_close = close0.fillna(close0.mean()).to_numpy()
        self.last_prices = init_close

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.date_memory = [self._day_date(self.day)]

        self.state = self._build_state_from_day_df(self._cur_day_df)
        return self.state

    def step(self, actions):
        # If at end, finish episode
        if self.day >= self.df.index.max():
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            table = print_metrics(OrderedDict({
                "Total Return": [f"{tr*100:0.4f}%"],
                "Sharp Ratio":  [f"{sharpe_ratio:0.4f}"],
                "Volatility":   [f"{vol*100:0.4f}%"],
                "Max Drawdown": [f"{mdd*100:0.4f}%"],
            }))
            df_value = self.save_asset_memory()
            assets = df_value["total assets"].values
            info = {"sharpe_ratio": sharpe_ratio, "total_assets": assets, "table": table}
            return self.state, 0.0, True, info

        # Convert action -> weights (cash + assets)
        actions = np.array(actions, dtype=float).reshape(-1)
        weights = self._softmax(actions)
        # Enforce shape
        if weights.shape[0] != self.action_space_shape:
            raise ValueError(f"Expected {self.action_space_shape} weights (cash+{self.stock_dim}), got {weights.shape[0]}")

        self.weights_memory.append(weights.tolist())

        # Advance day
        last_close = self.last_prices  # shape (stock_dim,)
        last_day_df = self._cur_day_df

        self.day += 1
        self._cur_day_df = self._day_slice(self.day)

        # Current closes aligned to universe; missing tickers -> use last_close (ratio 1.0)
        cur_close_raw = self._cur_day_df["close"].astype(float).to_numpy()
        cur_close = np.where(np.isnan(cur_close_raw), last_close, cur_close_raw)

        # Per-asset simple returns for the day
        price_rel = cur_close / (last_close + 1e-12)           # shape (stock_dim,)
        per_asset_ret = price_rel - 1.0

        # Portfolio return (exclude cash first element)
        portfolio_weights = weights[1:]
        portfolio_return = float(np.dot(per_asset_ret, portfolio_weights))

        # New weights after market movement (pre-rebalancing drift)
        drifted = np.array([weights[0]] + list(portfolio_weights * price_rel))
        weights_brandnew = self._normalize(drifted).tolist()
        self.weights_memory.append(weights_brandnew)

        # Transaction cost on weight change
        w_old = np.array(self.weights_memory[-3], dtype=float)
        w_new = np.array(self.weights_memory[-2], dtype=float)
        diff_weights = float(np.sum(np.abs(w_old - w_new)))
        fee = diff_weights * self.transaction_cost_pct * self.portfolio_value

        # Portfolio value update
        new_value = (self.portfolio_value - fee) * (1.0 + portfolio_return)
        reward = new_value - self.portfolio_value
        self.portfolio_value = new_value

        # Bookkeeping
        self.portfolio_return_memory.append((new_value / (self.asset_memory[-1] + 1e-12)) - 1.0)
        self.asset_memory.append(new_value)
        self.date_memory.append(self._day_date(self.day))
        self.transaction_cost_memory.append(fee)
        self.last_prices = cur_close  # update for next step

        # Next observation
        self.state = self._build_state_from_day_df(self._cur_day_df)

        return self.state, reward, False, {}

    # ---------- reporting ----------

    def save_portfolio_return_memory(self):
        df = pd.DataFrame({"date": self.date_memory, "daily_return": self.portfolio_return_memory})
        df.set_index("date", inplace=True)
        return df

    def save_asset_memory(self):
        df = pd.DataFrame({"date": self.date_memory, "total assets": self.asset_memory})
        df.set_index("date", inplace=True)
        return df

    def get_daily_return_rate(self, price_list: list[float]):
        return [(price_list[i + 1] / price_list[i]) - 1.0 for i in range(len(price_list) - 1)]

    def evaualte(self, df: pd.DataFrame):
        daily_return = df["daily_return"].values
        assets = df["total assets"].values
        tr = assets[-1] / (assets[0] + 1e-10) - 1.0

        # Risk metrics
        ret_rate = self.get_daily_return_rate(assets)
        sharpe = np.mean(ret_rate) * (252 ** 0.5) / (np.std(ret_rate) + 1e-10)
        vol = np.std(ret_rate)

        peak = assets[0]
        mdd = 0.0
        for v in assets:
            if v > peak:
                peak = v
            dd = (peak - v) / (peak + 1e-12)
            if dd > mdd:
                mdd = dd

        neg = daily_return[daily_return < 0]
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.nan_to_num(np.std(neg), nan=0.0) + 1e-10) / (np.sqrt(len(daily_return)) + 1e-10)
        return tr, sharpe, vol, mdd, cr, sor

    def analysis_result(self):
        df_return = self.save_portfolio_return_memory()
        df_value = self.save_asset_memory()
        df = pd.DataFrame({
            "daily_return": df_return["daily_return"].values,
            "total assets": df_value["total assets"].values,
        })
        return self.evaualte(df)
