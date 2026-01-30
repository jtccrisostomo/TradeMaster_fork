from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomDataset
from ..builder import DATASETS
from trademaster.utils import get_attr
import pandas as pd
import os
from typing import Optional, Set, List

ROOT = str(Path(__file__).resolve().parents[3])

def _abs_or_none(p: Optional[str]) -> Optional[str]:
    """Return absolute path if provided (relative paths are resolved to repo root)."""
    if not p:
        return None
    return p if osp.isabs(p) else osp.join(ROOT, p)

def _safe_unique_tics(path: Optional[str]) -> Set[str]:
    """Read unique tickers from a CSV without loading full dataset if possible."""
    if not path or not osp.exists(path):
        return set()
    try:
        # Fast path if 'tic' column is present
        df = pd.read_csv(path, usecols=["tic"])
    except Exception:
        df = pd.read_csv(path)
        if "tic" not in df.columns:
            raise RuntimeError(f"'tic' column not found in CSV: {path}")
    return set(map(str, df["tic"].unique()))

@DATASETS.register_module()
class PortfolioManagementDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(PortfolioManagementDataset, self).__init__()

        # Store raw kwargs for debugging/reproducibility
        self.kwargs = kwargs

        # Resolve data paths (train/valid/test)
        self.data_path  = _abs_or_none(get_attr(kwargs, "data_path", None))
        self.train_path = _abs_or_none(get_attr(kwargs, "train_path", None))
        self.valid_path = _abs_or_none(get_attr(kwargs, "valid_path", None))
        self.test_path  = _abs_or_none(get_attr(kwargs, "test_path",  None))

        # Optional dynamic test file (support both key spellings)
        tdp = (
            get_attr(kwargs, "test_dynamic_path", None)
            or get_attr(kwargs, "dynamics_test_path", None)
        )
        self.test_dynamic_path = _abs_or_none(tdp)

        # ---- Compute a consistent ticker universe (intersection) ----
        t_train = _safe_unique_tics(self.train_path)
        t_valid = _safe_unique_tics(self.valid_path)
        t_test  = _safe_unique_tics(self.test_path)

        non_empty: List[Set[str]] = [s for s in (t_train, t_valid, t_test) if len(s) > 0]
        if non_empty:
            common = set.intersection(*non_empty) if len(non_empty) > 1 else non_empty[0]
        else:
            common = set()

        # Fallback to train set if intersection is empty (unlikely if splits are consistent)
        if not common and t_train:
            common = t_train

        # Save a sorted, stable universe
        self.ticker_universe = sorted(common)

        # Core dataset parameters used by the environment
        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
        self.initial_amount = get_attr(kwargs, "initial_amount", 100000)
        self.length_day = get_attr(kwargs, "length_day", 10)
        self.transaction_cost_pct = get_attr(kwargs, "transaction_cost_pct", 0.001)
        # Optional fee model params (used by PSE fee schedule)
        self.fee_model = get_attr(kwargs, "fee_model", None)
        self.trade_weight_threshold = get_attr(kwargs, "trade_weight_threshold", 1e-3)
        self.trade_gross_threshold = get_attr(kwargs, "trade_gross_threshold", 100.0)
        self.pse_commission_rate = get_attr(kwargs, "pse_commission_rate", 0.0025)
        self.pse_commission_min = get_attr(kwargs, "pse_commission_min", 20.0)
        self.vat_rate = get_attr(kwargs, "vat_rate", 0.12)
        self.pse_fee_rate = get_attr(kwargs, "pse_fee_rate", 0.00005)
        self.sccp_fee_rate = get_attr(kwargs, "sccp_fee_rate", 0.0001)
        self.pse_fee_vat_cutoff = get_attr(kwargs, "pse_fee_vat_cutoff", "2025-09-04")
        self.stt_rate_pre = get_attr(kwargs, "stt_rate_pre", 0.006)
        self.stt_rate_post = get_attr(kwargs, "stt_rate_post", 0.001)
        self.stt_cutoff = get_attr(kwargs, "stt_cutoff", "2025-07-01")
        legacy_trades = get_attr(kwargs, "max_daily_trades", None)
        self.max_daily_turnover = get_attr(kwargs, "max_daily_turnover", legacy_trades)
        self.max_daily_trades = legacy_trades

    def get_styled_intervals_and_gives_new_index(self, data):
        index_by_tick_list = []
        index_by_tick = []
        date = data['date'].to_list()
        last_date = date[0]
        date_counter = 0
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                date_counter = -1
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
                index_by_tick_list.append(index_by_tick)
                index_by_tick = []
            if date[i] != last_date:
                date_counter += 1
            index_by_tick.append(date_counter)
            last_value = index[i]
            last_date = date[i]
        intervals.append([last_index, data.shape[0]])
        index_by_tick_list.append(index_by_tick)
        return intervals, index_by_tick_list
