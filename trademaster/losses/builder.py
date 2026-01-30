from trademaster.utils import build_from_cfg
from mmcv.utils import Registry
import copy

LOSSES = Registry('loss')

def build_loss(cfg, default_args=None):
    """Build loss function from cfg.loss using LOSSES registry."""
    cp_cfg = copy.deepcopy(cfg.loss)
    loss = build_from_cfg(cp_cfg, LOSSES, default_args)
    return loss
