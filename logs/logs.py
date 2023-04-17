import torch
import shutil
from pathlib import Path
from contextlib import contextmanager

from .base import LoggerBase, LoggerOnline
from .config import print_cfg, save_cfg
from .checkpoint import save_source_files
from .checkpoint import get_checkpoint_folder
from .tensorboard import tensorboard_logger_factory
from ..utils import pytorch_worker_info


online_loggers = {
    'tensorboard': tensorboard_logger_factory
}


class LoggerUnited(LoggerBase, LoggerOnline):
    """
    Main Logger in this library. Can work in multiprocessing mode.
    """
    def __init__(self, config, online_logger: str = None):
        """
        Function creates experiment directory and save main config files to it.

        Args:
            config: Environment config.yaml file
            online_logger (str): Tensorboard logger
        """
        self.checkpoint_folder = get_checkpoint_folder(config)
        super().__init__(__name__, path_target=self.checkpoint_folder)

        try:
            self.online_logger = online_loggers[online_logger](config)
            self.use_online = True
        except KeyError:
            self.online_logger = None
            self.use_online = False

        self._runs_stack = []

        self.log_gpu_stats()
        self.print_gpu_memory_usage()
        print_cfg(config, self)
        save_cfg(config, self)
        save_source_files(config, self)

    def get_exp_root_dir(self):
        return Path(self.checkpoint_folder)

    def set_run(self, run_name=None):
        if self.use_online:
            self.online_logger.set_run(run_name)

    @contextmanager
    def toggle_run(self, run_name=None):
        if self.use_online:
            with self.online_logger.toggle_run(run_name):
                yield
        else:
            yield

    def on_update(
        self,
        iteration=0,
        loss=None,
        log_frequency=None,
        batch_time=None,
        max_iteration=None
    ):
        """
        Use to log training process to online logger.

        Args:
            iteration (int): iteration from which log started
            loss (float): loss to log
            log_frequency (int): frequency (in epochs) to log (can be None)
            batch_time (List[float]): time execution for batches (can be None)
            max_iteration (int): Max iteration which to be logged (can be None)

        Returns:
            None:

        """
        if self.use_online:
            self.online_logger.on_update(
                iteration=iteration,
                loss=loss,
                log_frequency=log_frequency,
                batch_time=batch_time,
                max_iteration=max_iteration,
            )

    def log_metrics(self, tab=None, metrics=None, phase_idx=0):
        """
        Log metrics to online logger.

        Args:
            tab (str): table name
            metrics (dict): format to log metrics (e.g. {name: value})
            phase_idx (int): step index to bind log to

        Returns:
            None:

        """
        if self.use_online:
            rank, _, _, _ = pytorch_worker_info()
            if self.use_ddp and rank != 0:
                return
            self.online_logger.log_metrics(tab, metrics, phase_idx, rank)

    def log_scalars(self, tab=None, scalars=None, phase_idx=0):
        """
        Log scalars to online logger.

        Args:
            tab (str): table name
            scalars (dict): format to log metrics (e.g. {name: value})
            phase_idx (int): step index to bind log to

        Returns:
            None:

        """
        if self.use_online:
            self.online_logger.log_scalars(tab, scalars, phase_idx)

    def save_checkpoint(self, state, is_best, filename=None, only_main_rank=True):
        """
        Save state of model with label (best or no).

        Args:
            state (dict): model.state_dict
            is_best (bool): whether model is the best :)
            filename (str): filename to save checkpoint
            only_main_rank (bool): limit logging to rank 0 only

        Returns:
            None:

        """
        rank, _, _, _ = pytorch_worker_info()
        if only_main_rank and rank != 0:
            return

        if filename is None:
            filename = Path(self.checkpoint_folder) / "checkpoint.pth.tar"

        try:
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, filename.parent / "model_best.pth.tar")
        except Exception as e:
            print(f"Unable to save checkpoint to '{filename}' at this time due to {e}")

    # Wrappers for tensorboard functions
    def add_histogram(self, values=None, phase_idx=0, name="histogram"):
        if self.use_online:
            self.online_logger.add_histogram(
                values=values, phase_index=phase_idx, name=name
            )

    def add_embedding(self, embedding=None, tag=None, phase_index=None):
        if self.use_online:
            self.online_logger.add_embedding(embedding, tag, phase_index)

    def add_graph(self, model=None, inputs=None):
        if self.use_online:
            self.online_logger.add_graph(model, inputs)

    def add_images(self, tag=None, image=None):
        if self.use_online:
            self.online_logger.add_images(tag, image)

    def add_audio(self, tag=None, audio=None):
        if self.use_online:
            self.online_logger.add_images(tag, audio)