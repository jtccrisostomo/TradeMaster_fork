"""
Config for PSE Top-30 portfolio management using EIIE (FEES ENABLED).
"""

task_name = "portfolio_management"
dataset_name = "pse_top30_2008_2025"
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
    f"../_base_/transition/transition.py"
]

data = dict(
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
    transaction_cost_pct=0.0,  # Handled internally by the fee_model below
    fee_model="pse",           # Enabled PSE asymmetric tax/fee logic
    use_board_lot=True,        # Enabled realistic Philippine board lots
    trade_weight_threshold=1e-3,
    trade_gross_threshold=100.0
)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(type="Transition")

agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=5000,  
    gamma=0.9770288629446565,  
    policy_update_frequency=600
)  

trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=10,
    work_dir=work_dir,
    if_remove=False,
    batch_size=96,  
    horizon_len=512,  
    buffer_size=2500
) 

loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.0001113234498471106)  

act = dict(
    type="EIIEConv",
    input_dim=None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims=[48]  
)

cri = dict(
    type="EIIECritic",
    input_dim=None,
    action_dim=None,
    output_dim=1,
    time_steps=None,
    num_layers=1,
    hidden_size=64  
)
