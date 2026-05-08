"""
Config for PSE Top-30 portfolio management using EIIE.
This file is consumed by mmcv.Config and then expanded by replace_cfg_vals.

Key idea:
  - The backtest script will override data paths and some values at runtime.
  - The config defines defaults + model/trainer hyperparameters.
"""

task_name = "portfolio_management"
dataset_name = "pse_top30_2008_2025"
net_name = "eiie"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    # Base templates that define defaults for dataset/env/agent/trainer/etc.
    f"../_base_/datasets/{task_name}/dj30.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eiie_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

data = dict(
    # Dataset class + CSV paths (overridden by backtest script at runtime)
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/pse_top30_2008_2025',
    train_path='data/portfolio_management/pse_top30_2008_2025/train.csv',
    valid_path='data/portfolio_management/pse_top30_2008_2025/valid.csv',
    test_path='data/portfolio_management/pse_top30_2008_2025/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.0,  # frictionless per latest request
    # Disable PSE-specific fees/rounding
    fee_model=None,
    use_board_lot=False,
    trade_weight_threshold=1e-3,
    trade_gross_threshold=100.0)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(
    type = "Transition"
)
agent = dict(
    # EIIE agent hyperparameters (affects replay + policy update cadence)
    type='PortfolioManagementEIIE',
    memory_capacity=5000,  # tuned
    gamma=0.9770288629446565,  # tuned
    policy_update_frequency=600)  # tuned

trainer = dict(
    # Trainer settings for off-policy replay buffer
    type='PortfolioManagementEIIETrainer',
    epochs=10,
    work_dir=work_dir,
    if_remove=False,
    batch_size=96,  # tuned
    horizon_len=512,  # tuned
    buffer_size=2500)  # raised relative to horizon_len to avoid sampling underflow

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.0001113234498471106)  # tuned

act = dict(
    # Policy network config (updated at runtime with input_dim/time_steps)
    type = "EIIEConv",
    input_dim = None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims = [48]  # tuned
)

cri = dict(
    # Critic network config (updated at runtime with input_dim/action_dim/time_steps)
    type = "EIIECritic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=64  # tuned
)
