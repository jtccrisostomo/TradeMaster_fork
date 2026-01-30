from mmcv.utils import Registry
from trademaster.utils import build_from_cfg

TRAINERS = Registry('trainer')

def build_trainer(cfg, default_args=None):
    """Build trainer instance from cfg.trainer using TRAINERS registry."""
    cp_cfg = dict(cfg.trainer)
    trainer = build_from_cfg(cp_cfg, TRAINERS, default_args)
    return trainer
