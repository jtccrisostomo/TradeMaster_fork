from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

DATASETS = Registry('dataset')

def build_dataset(cfg, default_args=None):
    """Build dataset instance from cfg.data using the DATASETS registry."""
    cp_cfg = copy.deepcopy(cfg.data)
    dataset = build_from_cfg(cp_cfg, DATASETS, default_args)
    return dataset
