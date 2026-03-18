data = dict(
    type='PortfolioManagementDataset',
    data_path=
    '/mnt/lustre/erdtlstr/home/laperia/TradeMaster/data/portfolio_management/nyse_ohlc',
    train_path=
    '/mnt/lustre/erdtlstr/home/laperia/TradeMaster/data/portfolio_management/nyse_ohlc/train.csv',
    valid_path=
    '/mnt/lustre/erdtlstr/home/laperia/TradeMaster/data/portfolio_management/nyse_ohlc/valid.csv',
    test_path=
    '/mnt/lustre/erdtlstr/home/laperia/TradeMaster/data/portfolio_management/nyse_ohlc/test.csv',
    train_start_date='2010-01-01',
    train_end_date='2017-12-31',
    valid_start_date='2018-01-01',
    valid_end_date='2018-12-31',
    test_start_date='2019-01-01',
    test_end_date='2024-12-31',
    initial_amount=100000,
    transaction_cost_pct=0.001,
    tech_indicator_list=['open', 'high', 'low', 'close', 'volume'],
    length_day=10,
    test_dynamic='-1')
environment = dict(type='PortfolioManagementEnvironment')
trainer = dict(
    type='PortfolioManagementTrainer',
    agent_name='ppo',
    if_remove=False,
    configs=dict(framework='torch', num_workers=0),
    work_dir='work_dir/portfolio_management_dj30_ppo_ppo_adam_mse_fg',
    epochs=2,
    device='cuda',
    seed=1,
    ray_kwargs=dict(address=None))
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
_CWD = '/mnt/lustre/erdtlstr/home/laperia/TradeMaster'
_DATA_ROOT = '/mnt/lustre/erdtlstr/home/laperia/TradeMaster/data'
task_name = 'portfolio_management'
dataset_name = 'dj30'
net_name = 'ppo'
agent_name = 'ppo'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_dj30_ppo_ppo_adam_mse_fg'
seed = 1
