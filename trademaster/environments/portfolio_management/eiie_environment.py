from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from trademaster.utils import get_attr, print_metrics
import pandas as pd
from ..custom import Environments
from ..builder import ENVIRONMENTS
from gym import spaces
from collections import OrderedDict
import pickle
import os.path as osp


@ENVIRONMENTS.register_module()
class PortfolioManagementEIIEEnvironment(Environments):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIEEnvironment, self).__init__()

        # Dataset and task split (train/valid/test)
        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.test_dynamic=int(get_attr(kwargs, "test_dynamic", "-1"))
        self.task_index = int(get_attr(kwargs, "task_index", "-1"))
        self.work_dir = get_attr(kwargs, "work_dir", "")
        time_steps = get_attr(self.dataset, "time_steps", 10)
        self.day = time_steps - 1

        # Select the CSV path based on task
        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)



        # Portfolio/account parameters
        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.fee_model = get_attr(self.dataset, "fee_model", None)
        self.trade_weight_threshold = float(get_attr(self.dataset, "trade_weight_threshold", 1e-3))
        self.trade_gross_threshold = float(get_attr(self.dataset, "trade_gross_threshold", 100.0))
        self.pse_commission_rate = float(get_attr(self.dataset, "pse_commission_rate", 0.0025))
        self.pse_commission_min = float(get_attr(self.dataset, "pse_commission_min", 20.0))
        self.vat_rate = float(get_attr(self.dataset, "vat_rate", 0.12))
        self.pse_fee_rate = float(get_attr(self.dataset, "pse_fee_rate", 0.00005))
        self.sccp_fee_rate = float(get_attr(self.dataset, "sccp_fee_rate", 0.0001))
        self.pse_fee_vat_rate = float(get_attr(self.dataset, "pse_fee_vat_rate", self.vat_rate))
        self.pse_fee_vat_cutoff = pd.Timestamp(
            get_attr(self.dataset, "pse_fee_vat_cutoff", "2025-09-04")
        )
        self.stt_rate_pre = float(get_attr(self.dataset, "stt_rate_pre", 0.006))
        self.stt_rate_post = float(get_attr(self.dataset, "stt_rate_post", 0.001))
        self.stt_cutoff = pd.Timestamp(get_attr(self.dataset, "stt_cutoff", "2025-07-01"))
        self.max_daily_turnover = get_attr(self.dataset, "max_daily_turnover", None)
        self.max_daily_trades = get_attr(self.dataset, "max_daily_trades", None)
        # Tech indicator list defines state channels (features used in state tensor)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        # Load data for the selected task
        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date = self.df.loc[:, 'date'].iloc[0]
            self.end_date = self.df.loc[:, 'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)

        # Action/state shapes
        self.stock_dim = len(self.df.tic.unique())
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim + 1
        self.time_steps = time_steps

        self.action_space = spaces.Box(low=-5,
                                       high=5,
                                       shape=(self.action_space_shape,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list),
                   self.state_space_shape,
                   self.time_steps))

        self.action_dim = self.stock_dim
        self.state_dim = self.observation_space.shape[0]

        # Initial state window (time_steps days ending at current day)
        self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :]
        self.state = np.array([[
            self.data[self.data.tic == tic][tech].values.tolist()
            for tech in self.tech_indicator_list
        ] for tic in self.data.tic.unique()])
        self.state = np.transpose(self.state, (0, 2, 1))

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / (self.stock_dim + 1)] *
                               (self.stock_dim + 1)]
        self.raw_weights_memory = []
        self.executed_weights_memory = []
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        self.current_trade_date = self.date_memory[0]
        self.daily_turnover_used = 0.0
        self.daily_turnover_summary = OrderedDict()
        self.daily_trade_count = 0
        self.daily_trade_summary = OrderedDict()
        self.transaction_fee_detail_memory = []
        self.test_id = 'agent'

    def reset(self):
        """Reset environment to the initial day and rebuild the state window."""
        self.day = self.time_steps - 1
        self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([[
            self.data[self.data.tic == tic][tech].values.tolist()
            for tech in self.tech_indicator_list
        ] for tic in self.data.tic.unique()])
        self.state = np.transpose(self.state, (0, 2, 1))
        # self.state = np.transpose(self.state, (2, 0, 1))
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / (self.stock_dim + 1)] *
                               (self.stock_dim + 1)]
        self.raw_weights_memory = []
        self.executed_weights_memory = []
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        self.current_trade_date = self.date_memory[0]
        self.daily_turnover_used = 0.0
        self.daily_turnover_summary = OrderedDict()
        self.daily_trade_count = 0
        self.daily_trade_summary = OrderedDict()
        self.transaction_fee_detail_memory = []

        return self.state

    def step(self, weights):
        """
        Execute one trading step.
        weights: portfolio weights including cash (length = stock_dim + 1).
        """
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        weights = np.array(weights, dtype=float).flatten()
        if weights.shape[0] != self.stock_dim + 1:
            raise ValueError(
                f"Expected {self.stock_dim + 1} weights (cash + assets), got {weights.shape[0]}"
            )
        weights = weights / (np.sum(weights) + 1e-12)

        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format(tr * 100)],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol* 100)],
                    "Max Drawdown": ["{:04f}%".format(mdd* 100)],
                    # "Calmar Ratio": ["{:04f}".format(cr)],
                    # "Sortino Ratio": ["{:04f}".format(sor)],
                }
            )
            table = print_metrics(stats)
            print(table)

            # Save metrics when episode ends
            df_return = self.save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df_value = self.save_asset_memory()
            assets = df_value["total assets"].values
            #TODO calculate the buy and hold
            save_dict = OrderedDict(
                {
                    "Profit Margin": tr * 100,
                    "Excess Profit": tr * 100-0,
                    "daily_return": daily_return,
                    "total_assets": assets
                }
            )
            metric_save_path=osp.join(self.work_dir,'metric_'+str(self.task)+'_'+str(self.test_dynamic)+'_'+str(self.test_id)+'_'+str(self.task_index)+'.pickle')
            if self.task == 'test_dynamic':
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self.state, 0, self.terminal, {"sharpe_ratio": sharpe_ratio,"total_assets": assets}

        else:  # directly use the process of
            self.raw_weights_memory.append(weights.tolist())
            last_day_memory = self.df.loc[self.day, :]
            self.day += 1
            self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :]
            self.state = np.array([[
                self.data[self.data.tic == tic][tech].values.tolist()
                for tech in self.tech_indicator_list
            ] for tic in self.data.tic.unique()])
            self.state = np.transpose(self.state, (0, 2, 1))

            # self.state = np.transpose(self.state, (2, 0, 1))
            new_price_memory = self.df.loc[self.day, :]
            # Price relative between today and yesterday (used for portfolio return)
            price_rel = (
                new_price_memory.close.values / last_day_memory.close.values
            )

            weights_old_np = np.array(self.weights_memory[-1], dtype=float)

            trade_date = self.df.loc[self.day, "date"]
            if isinstance(trade_date, (pd.Series, np.ndarray)):
                trade_date = (
                    trade_date.iloc[0]
                    if hasattr(trade_date, "iloc")
                    else trade_date[0]
                )
            # Reset daily turnover/trade counters on date change
            if trade_date != self.current_trade_date:
                self.current_trade_date = trade_date
                self.daily_turnover_used = 0.0
                self.daily_trade_count = 0

            diff_assets = weights[1:] - weights_old_np[1:]
            # Turnover: sum of absolute weight changes
            turnover = float(np.sum(np.abs(diff_assets)))
            if self.max_daily_turnover is not None:
                quota = self.max_daily_turnover - self.daily_turnover_used
                if quota <= 0:
                    diff_assets[:] = 0.0
                elif turnover > quota:
                    diff_assets *= quota / (turnover + 1e-12)

            # Trade count constraint (limit number of tickers traded per day)
            trade_mask = np.abs(diff_assets) > 1e-6
            trade_units = int(np.sum(trade_mask))
            if self.max_daily_trades is not None:
                quota_trades = self.max_daily_trades - self.daily_trade_count
                if quota_trades <= 0:
                    diff_assets[:] = 0.0
                    trade_units = 0
                elif trade_units > quota_trades:
                    idx = np.argsort(np.abs(diff_assets))
                    keep = idx[-quota_trades:] if quota_trades > 0 else []
                    mask = np.zeros_like(diff_assets, dtype=bool)
                    mask[keep] = True
                    diff_assets = np.where(mask, diff_assets, 0.0)
                    trade_units = quota_trades if quota_trades > 0 else 0
                self.daily_trade_count += trade_units

            turnover = float(np.sum(np.abs(diff_assets)))
            if self.max_daily_turnover is not None:
                self.daily_turnover_used += turnover

            weights_exec_assets = weights_old_np[1:] + diff_assets
            weights_exec_assets = np.clip(weights_exec_assets, 0.0, None)
            asset_sum = float(np.sum(weights_exec_assets))
            cash = 1.0 - asset_sum
            if cash < 0.0:
                weights_exec_assets = weights_exec_assets / (asset_sum + 1e-12)
                cash = 0.0
            weights_exec = np.concatenate(([cash], weights_exec_assets))
            self.executed_weights_memory.append(weights_exec.tolist())
            key = str(self.current_trade_date)
            if turnover > 0:
                self.daily_turnover_summary[key] = (
                    self.daily_turnover_summary.get(key, 0.0) + turnover
                )
            if trade_units > 0:
                self.daily_trade_summary[key] = (
                    self.daily_trade_summary.get(key, 0) + trade_units
                )

            if self.fee_model == "pse":
                transcationfee = self._pse_transaction_fee(
                    weights_old_np,
                    weights_exec,
                    self.portfolio_value,
                    trade_date,
                )
            else:
                diff_weights = np.sum(
                    np.abs(weights_exec_assets - weights_old_np[1:])
                )
                transcationfee = (
                    diff_weights * self.transaction_cost_pct * self.portfolio_value
                )
            portfolio_return = float(
                np.sum((price_rel - 1) * weights_exec_assets)
            )
            new_portfolio_value = (self.portfolio_value -
                                   transcationfee) * (1 + portfolio_return)
            portfolio_return = (new_portfolio_value -
                                self.portfolio_value) / self.portfolio_value
            self.reward = np.log(new_portfolio_value) - np.log(
                self.portfolio_value)
            self.portfolio_value = new_portfolio_value

            weights_brandnew = self.normalization(
                [cash] + list(weights_exec_assets * price_rel)
            )
            self.weights_memory.append(weights_brandnew.tolist())
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[-1])
            self.asset_memory.append(new_portfolio_value)

        return self.state, self.reward, self.terminal, {"weights_brandnew":weights_brandnew}

    def _pse_transaction_fee(
        self,
        weights_old: np.ndarray,
        weights_new: np.ndarray,
        portfolio_value: float,
        trade_date,
    ) -> float:
        asset_diff = weights_new[1:] - weights_old[1:]
        abs_diff = np.abs(asset_diff)
        gross = abs_diff * portfolio_value
        trade_mask = (abs_diff >= self.trade_weight_threshold) & (
            gross >= self.trade_gross_threshold
        )

        buy_gross = np.where((asset_diff > 0) & trade_mask, gross, 0.0)
        sell_gross = np.where((asset_diff < 0) & trade_mask, gross, 0.0)

        stt_rate = self._stt_rate(trade_date)
        pse_vat_rate = self._pse_fee_vat_rate(trade_date)
        buy_detail = self._pse_side_fee_detail(buy_gross, stt_rate=0.0, pse_vat_rate=pse_vat_rate)
        sell_detail = self._pse_side_fee_detail(sell_gross, stt_rate=stt_rate, pse_vat_rate=pse_vat_rate)
        total_fee = buy_detail["total_fee"] + sell_detail["total_fee"]
        self.transaction_fee_detail_memory.append(
            {
                "date": str(trade_date),
                "buy_gross": buy_detail["gross"],
                "sell_gross": sell_detail["gross"],
                "commission": buy_detail["commission"] + sell_detail["commission"],
                "vat": buy_detail["vat"] + sell_detail["vat"],
                "pse_fee": buy_detail["pse_fee"] + sell_detail["pse_fee"],
                "pse_fee_vat": buy_detail["pse_fee_vat"] + sell_detail["pse_fee_vat"],
                "sccp_fee": buy_detail["sccp_fee"] + sell_detail["sccp_fee"],
                "stt_fee": buy_detail["stt_fee"] + sell_detail["stt_fee"],
                "total_fee": total_fee,
                "buy_trades": buy_detail["trade_count"],
                "sell_trades": sell_detail["trade_count"],
            }
        )
        return float(total_fee)

    def _pse_side_fee_detail(
        self,
        gross: np.ndarray,
        stt_rate: float,
        pse_vat_rate: float,
    ) -> dict:
        has_trade = gross > 0
        commission = np.where(
            has_trade,
            np.maximum(self.pse_commission_rate * gross, self.pse_commission_min),
            0.0,
        )
        vat = commission * self.vat_rate
        pse_fee = gross * self.pse_fee_rate
        pse_fee_vat = pse_fee * pse_vat_rate
        sccp_fee = gross * self.sccp_fee_rate
        stt_fee = gross * stt_rate
        return {
            "gross": float(np.sum(gross)),
            "commission": float(np.sum(commission)),
            "vat": float(np.sum(vat)),
            "pse_fee": float(np.sum(pse_fee)),
            "pse_fee_vat": float(np.sum(pse_fee_vat)),
            "sccp_fee": float(np.sum(sccp_fee)),
            "stt_fee": float(np.sum(stt_fee)),
            "total_fee": float(np.sum(commission + vat + pse_fee + pse_fee_vat + sccp_fee + stt_fee)),
            "trade_count": int(np.sum(has_trade)),
        }

    def _stt_rate(self, trade_date) -> float:
        if trade_date is None:
            return self.stt_rate_pre
        try:
            trade_ts = pd.to_datetime(trade_date)
        except (ValueError, TypeError):
            return self.stt_rate_pre
        return self.stt_rate_post if trade_ts >= self.stt_cutoff else self.stt_rate_pre

    def _pse_fee_vat_rate(self, trade_date) -> float:
        if trade_date is None:
            return 0.0
        try:
            trade_ts = pd.to_datetime(trade_date)
        except (ValueError, TypeError):
            return 0.0
        return self.pse_fee_vat_rate if trade_ts >= self.pse_fee_vat_cutoff else 0.0

    def save_fee_memory(self):
        if not self.transaction_fee_detail_memory:
            return pd.DataFrame()
        return pd.DataFrame(self.transaction_fee_detail_memory)

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.sum(actions)
        actions = actions / sum
        return actions

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)

    def get_daily_return_rate(self,price_list:list):
        return_rate_list=[]
        for i in range(len(price_list)-1):
            return_rate=(price_list[i+1]/price_list[i])-1
            return_rate_list.append(return_rate)
        return return_rate_list
        

    def evaualte(self, df):
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        return_rate_list=self.get_daily_return_rate(df["total assets"].values)

        sharpe_ratio = np.mean(return_rate_list)*(252)** 0.5 / (np.std(return_rate_list) + 1e-10)
        vol = np.std(return_rate_list)
        mdd = 0
        peak=df["total assets"][0]
        for value in df["total assets"]:
            if value>peak:
                peak=value
            dd=(peak-value)/peak
            if dd>mdd:
                mdd=dd
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.nan_to_num(np.std(neg_ret_lst), nan=0.0) + 1e-10) / (np.sqrt(len(daily_return))+1e-10)
        return tr, sharpe_ratio, vol, mdd, cr, sor
