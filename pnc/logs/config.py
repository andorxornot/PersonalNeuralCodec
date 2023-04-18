from pprint import pformat

from omegaconf import DictConfig, OmegaConf

from .base import LoggerBase, MessageType
from .checkpoint import get_checkpoint_folder
from ..utils import pytorch_worker_info


def print_cfg(cfg: DictConfig, logger: LoggerBase):
    """
    Supports printing both DictConfig and also the AttrDict config

    """
    logger.log("Training with config:", MessageType.INFO, main_rank_only=True)
    if isinstance(cfg, DictConfig):
        if hasattr(cfg, "pretty"):
            # Backward compatibility
            logger.log(cfg.pretty(), MessageType.INFO, main_rank_only=True)
        else:
            # Newest version of OmegaConf
            logger.log(OmegaConf.to_yaml(cfg), MessageType.INFO,
                       main_rank_only=True)
    else:
        logger.log(pformat(cfg), MessageType.WARNING, main_rank_only=True)


def save_cfg(cfg: DictConfig, logger: LoggerBase):
    """
    Saves cfg to experiment folder

    :param cfg: Environment config.yaml file
    :param logger: logger
    """
    rank, _, _, _ = pytorch_worker_info()
    checkpoint_folder = get_checkpoint_folder(cfg)
    logger.log(
        f"Saving config to {checkpoint_folder}/config.yaml:",
        MessageType.INFO,
        main_rank_only=True,
    )
    if isinstance(cfg, DictConfig) and rank == 0:
        OmegaConf.save(config=cfg, f=f"{checkpoint_folder}/config.yaml")
    else:
        logger.log(
            f"Config is not of 'DictConfig' type!",
            MessageType.ERROR,
            main_rank_only=True
        )
