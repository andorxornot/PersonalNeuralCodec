from glob import glob
from os import makedirs, path as osp, walk
from shutil import copy2 as copy

from .base import LoggerBase, MessageType
from ..utils import pytorch_worker_info


PREFIX_SOURCE = 'project_source'


def get_checkpoint_folder(config):
    """
    Check, create and return the checkpoint folder. User can specify their own
    checkpoint directory otherwise the default "." is used.
    """
    path_target = config.EXPERIMENT.DIR

    makedirs(path_target, exist_ok=True)
    assert osp.exists(
        config.EXPERIMENT.DIR
    ), (
        f"Please specify valid 'config.CHECKPOINT.DIR' directory."
        f" Failed: {config.CHECKPOINT.DIR}"
    )
    return path_target


def save_source_files(config, logger: LoggerBase):
    rank, _, _, _ = pytorch_worker_info()
    checkpoint_folder = get_checkpoint_folder(config)
    logger.log(
        f"Saving source files to {checkpoint_folder}/{PREFIX_SOURCE}",
        MessageType.INFO,
        main_rank_only=True,
    )
    assert osp.exists(
        config.PROJECT.ROOT
    ), (
        f"Please specify  valid 'config.PROJECT.ROOT' directory."
        f" Failed: {config.PROJECT.ROOT}"
    )
    directories_source = [
        osp.join(config.PROJECT.ROOT, x) for x in config.PROJECT.DEFAULT_DIRS
    ]
    pyfiles = glob(osp.join(config.PROJECT.ROOT, '*.py'))
    for directory in directories_source:
        pyfiles.extend([
            y for x in walk(directory)
            for y in glob(osp.join(x[0].decode(encoding='utf-8'), '*.py'))
        ])
    for pyfile in pyfiles:
        source = pyfile
        target = osp.join(
            f"{checkpoint_folder}", f"{PREFIX_SOURCE}",
            f'{pyfile.replace(config.PROJECT.ROOT, "")}',
        )
        path_target = osp.basename(target)
        if rank == 0:
            makedirs(path_target, exist_ok=True)
            copy(source, target)


__all__ = ['get_checkpoint_folder', 'save_source_files']
