from glob import glob
from os import makedirs, path as osp, walk
from shutil import copy2 as copy

from .base import LoggerBase, MessageType
from ..utils import pytorch_worker_info


# prefix_sources = 'project_source'


def get_checkpoint_folder(config):
    """
    Check, create and return the checkpoint folder. Default path in config:
    <experiment.path_output>/<experiment.name>/<experiment.path_checkpoints>
    """
    path_checkpoints = osp.join(config.experiment.path_output,
                                config.experiment.name,
                                config.experiment.path_checkpoints)

    makedirs(path_checkpoints, exist_ok=True)
    assert osp.exists(
        path_checkpoints
    ), (
        f"Please specify valid experiment directories."
        f" Failed to create: {path_checkpoints}."
        f" Refer to doc:\n{get_checkpoint_folder.__doc__}"
    )
    return path_checkpoints


def save_source_files(config, logger: LoggerBase):
    rank, _, _, _ = pytorch_worker_info()
    path_checkpoints = get_checkpoint_folder(config)
    prefix_sources = config.experiment.path_checkpoint_sources
    # path_checkpoints = osp.join(path_checkpoints, prefix_sources)
    logger.log(
        f"Saving source files to {osp.join(path_checkpoints, prefix_sources)}",
        MessageType.INFO,
        main_rank_only=True,
    )
    path_project_root = osp.realpath(config.experiment.path_project_root)
    assert osp.exists(
        path_project_root
    ), (
        f"Please specify  valid 'config.experiment.path_project_root'"
        f" directory. Failed: '{path_project_root}'"
    )

    # Get real directories from <experiment.project_checkpoint_directories>.
    # Exclude <experiment.path_output> patterns from the list
    directories_source = [
        osp.realpath(osp.join(path_project_root, x))
        for x in config.experiment.project_checkpoint_directories
        if x not in config.experiment.path_output
    ]
    # Get all Python sources from the <path_project_root>
    pyfiles = glob(osp.join(path_project_root, '*.py'))
    # Append Python sources from the <project_checkpoint_directories>
    for directory in directories_source:
        pyfiles.extend([
            osp.realpath(y) for x in walk(directory)
            for y in glob(osp.join(x[0].decode(encoding='utf-8'), '*.py'))
        ])
    # Replace source prefix with target (checkpoints) prefix and copy
    for pyfile in pyfiles:
        source = pyfile
        target = osp.join(
            f"{path_checkpoints}", f"{prefix_sources}",
            f"{pyfile.replace(path_project_root, '')}".lstrip(osp.sep)
        )
        path_target_directory = osp.basename(target)
        if rank == 0:
            makedirs(path_target_directory, exist_ok=True)
            copy(source, target)


__all__ = ['get_checkpoint_folder', 'save_source_files']
