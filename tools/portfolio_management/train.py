import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import os
import os.path as osp
import argparse
import torch
from mmcv import Config

# Repo root: TradeMaster/
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trademaster.utils import replace_cfg_vals, set_seed
from trademaster.datasets.builder import build_dataset
from trademaster.trainers.builder import build_trainer

set_seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="Run portfolio-management training/testing")
    # Point the default to your custom config (you can still override via --config)
    parser.add_argument(
        "--config",
        default=osp.join(ROOT, "configs", "portfolio_management", "PPO_nyse_custom.py"),
        help="Path to config file",
    )
    parser.add_argument("--task_name", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--test_dynamic", type=str, default="-1")
    parser.add_argument("--verbose", type=int, default=1)  # int default (not string)
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    # Expose test_dynamic flag (dataset may ignore; harmless to include)
    cfg.data.update({"test_dynamic": args.test_dynamic})

    if args.verbose:
        print(f"DEBUG: sarl_trainer.py - RAY_ADDRESS from env: {os.getenv('RAY_ADDRESS')}")
        print(f"Config (path: {args.config}): {cfg}")

    # Build dataset first (trainer expects it)
    dataset = build_dataset(cfg)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Work dir (absolute)
    work_dir = osp.join(ROOT, cfg.trainer.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    # Save the resolved config alongside the run
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    # Build trainer (passes dataset + device into trainer ctor)
    trainer = build_trainer(cfg, default_args=dict(dataset=dataset, device=device))

    if args.task_name.startswith("train"):
        trainer.train_and_valid()
        print("train end")
    elif args.task_name.startswith("test"):
        trainer.test()
        print("test end")

if __name__ == "__main__":
    main()
    """
    algorithmic_trading
    portfolio_management
    order_execution
    """
