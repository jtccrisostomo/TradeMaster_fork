"""
Config for NYSE portfolio management using EIIE (FEES ENABLED).
"""

task_name = "portfolio_management"
dataset_name = "nyse_ohlc_aligned30_2008_2025"
net_name = "eiie"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_fees"

_base_ = [
    f"../_base_/datasets/{task_name}/dj30.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eiie_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py",
]

data = dict(
    type="PortfolioManagementDataset",
    data_path="data/portfolio_management/nyse_ohlc_aligned30_2008_2025",
    train_path="data/portfolio_management/nyse_ohlc_aligned30_2008_2025/train.csv",
    valid_path="data/portfolio_management/nyse_ohlc_aligned30_2008_2025/valid.csv",
    test_path="data/portfolio_management/nyse_ohlc_aligned30_2008_2025/test.csv",
    tech_indicator_list=[
        "zopen", "zhigh", "zlow", "zadjcp", "zclose",
        "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30",
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,  # Applied standard 0.1% transaction cost
    fee_model=None,              # Explicitly disabled
    use_board_lot=False,         # Explicitly disabled
)

environment = dict(type="PortfolioManagementEIIEEnvironment")
transition = dict(type="Transition")

agent = dict(
    type="PortfolioManagementEIIE",
    memory_capacity=500,  
    gamma=0.969522982710727,  
    policy_update_frequency=300,  
)

trainer = dict(
    type="PortfolioManagementEIIETrainer",
    epochs=10,
    work_dir=work_dir,
    if_remove=False,
    batch_size=96,  
    horizon_len=256,  
    buffer_size=1500,  
)

loss = dict(type="MSELoss")
optimizer = dict(type="Adam", lr=5.1809895833717616e-05)  

act = dict(
    type="EIIEConv",
    input_dim=None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims=[128],  
)

cri = dict(
    type="EIIECritic",
    input_dim=None,
    action_dim=None,
    output_dim=1,
    time_steps=None,
    num_layers=1,
    hidden_size=48,  
)