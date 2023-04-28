import logging

from contextlib import contextmanager
from dataclasses import dataclass
from os import path as osp

import torch

from .base import LoggerOnline, get_path_logs
from .io import is_file_empty, makedirs


def tensorboard_available():
    """
    Check whether tensorboard is available or not.
    Returns:
        tb_available (bool): based on tensorboard imports, returns whether
                             TensorBoard is available or not.
    """
    try:
        import tensorboard  # noqa F401
        from torch.utils.tensorboard import SummaryWriter  # noqa F401

        tb_available = True
    except ImportError:
        logging.warning("Tensorboard is not available")
        tb_available = False
    return tb_available


def tensorboard_paths(cfg):
    """
    Get the output directory where the tensorboard events will be written.
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        Tensorboard as well as log directory, logging frequency etc.
    Returns:
        path_tensorboard (str): tensorboard output directory path
    """
    path_logs = get_path_logs(cfg)
    path_tensorboard = osp.join(path_logs, cfg.experiment.path_logs_tensorboard)
    logging.info(f"Tensorboard path = {path_tensorboard}")  # FIXME: to logger
    makedirs(path_tensorboard, exist_ok=True)

    path_tensorboard_csv = osp.join(path_logs,
                                    cfg.experiment.path_logs_tensorboard, 'csv')
    logging.info(f"Tensorboard csv path = {path_tensorboard_csv}")  # FIXME: ^
    makedirs(path_tensorboard_csv, exist_ok=True)
    return path_tensorboard, path_tensorboard_csv


def tensorboard_logger_factory(cfg):
    """
    Construct the Tensorboard logger for visualization from the specified config
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well as log directory, logging frequency etc.
    Returns:
        TensorboardHook (function): the tensorboard logger constructed
    """

    # Get the tensorboard directory and check tensorboard is installed
    path_tensorboard, path_tensorboard_csv = tensorboard_paths(cfg)
    # flush_secs = cfg.TENSORBOARD_SETUP.FLUSH_EVERY_N_MIN * 60
    flush_secs = cfg.experiment.log_tensorboard_flush_frequency * 60
    return LoggerTensorboard(
        path_base=path_tensorboard,
        path_csv=path_tensorboard_csv,
        flush_secs=flush_secs,
        # log_params=cfg.experiment.log_tensorboard_params,
        log_params=cfg.experiment.log_tensorboard_params,
        # log_params_frequency=cfg.TENSORBOARD_SETUP.LOG_PARAMS_EVERY_N_ITERS,
        log_params_frequency=cfg.experiment.log_tensorboard_params_frequency
    )


BYTE_TO_MiB = 2 ** 20


@dataclass
class Tabs:
    TRAIN = 'Training'
    VALID = 'Validation'
    TEST = 'Test'
    HIST = 'Histograms'
    EMBED = 'Embeddings'
    GRAPH = 'Graphs'
    IMAGE = 'Images'
    AUDIO = 'Audio'
    SPEED = 'Performance'
    MEMORY = 'Memory'


class LoggerTensorboard(LoggerOnline):
    """
    Tensorboard logger
    """
    def __init__(
        self,
        path_base,
        path_csv,
        flush_secs,
        log_params=False,
        log_params_frequency=-1
    ):
        """
        The constructor method of LoggerTensorboard.

        Args:
            path_base (str): Tensorboard base directory
            path_csv (str): directory path to store metrics as csv files
            log_params (bool): whether to log model params to tensorboard
            log_params_frequency (int): frequency at which parameters should be
                                        logged to tensorboard (every N iterations)
        """
        if not tensorboard_available():
            raise RuntimeError(
                "tensorboard not installed, cannot use 'LoggerTensorboard'"
            )

        from torch.utils.tensorboard import SummaryWriter

        self.tb_class = SummaryWriter
        logging.info("Setting up Tensorboard Logger...")  # TODO: to logger
        self.path_base = path_base
        self.path_csv = path_csv
        self.cur_run = None
        self._runs_stack = []
        self.flush_secs = flush_secs
        self.writer = self.tb_class(log_dir=path_base, flush_secs=flush_secs)
        self.log_params = log_params
        self.log_params_every_n_iterations = log_params_frequency

        logging.info(
            f"Tensorboard config: log_params = {self.log_params}, "
            f"log_params_frequency = {self.log_params_every_n_iterations}"
        )  # TODO: to logger

    def set_run(self, run_name=None):
        log_dir = self.path_base
        if run_name is not None:
            log_dir += f"/{run_name}"
        self.writer.close()
        self.writer = self.tb_class(log_dir=log_dir, flush_secs=self.flush_secs)

        if run_name is not None:
            self.path_csv += f"/{run_name}"
        makedirs(self.path_csv, exist_ok=True)

    @contextmanager
    def toggle_run(self, run_name=None):
        self._runs_stack.append(self.cur_run)
        self.set_run(run_name)
        yield
        prev_run = self._runs_stack.pop()
        self.set_run(prev_run)

    def log_scalars(self, tab=None, scalars=None, phase_index=0):
        tab = tab or Tabs.TRAIN
        scalars = scalars or {}
        name = tab.replace('/', '_')  # "_".join(tab.split("/"))
        csv_name = osp.join(f"{self.path_csv}", f"{name}_scalars.csv")
        csv = open(csv_name, "a+")
        if is_file_empty(csv_name):
            csv.write(f"phase_index,{','.join(scalars.keys())}\n")
        # Log train/test accuracy
        metrics_string = f"{phase_index}"
        self.writer.add_scalars(
            tab,
            scalars,
            global_step=phase_index,
        )

        for k, score in scalars.items():
            metrics_string += f",{score}"
            # Reset the GPU Memory counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # torch.cuda.reset_max_memory_cached()
        metrics_string += "\n"
        csv.write(metrics_string)
        csv.close()

    def log_metrics(self, tab=None, metrics=None, phase_index=0, rank=0):
        """
        Log some arbitrary metrics. Also resets the CUDA memory counter.
        """
        tab = tab or Tabs.TRAIN
        metrics = metrics or {}
        csv_name = f"{self.path_csv}/{tab}_metrics.csv"
        csv = open(csv_name, "a+")
        if is_file_empty(csv_name):
            csv.write(f"phase_index,process,{','.join(metrics.keys())}\n")
        # Log train/test accuracy
        metrics_string = f"{phase_index},{rank}"

        for metric_name, score in metrics.items():
            t = f"{tab}/Process {rank}/{metric_name}"
            self.writer.add_scalar(
                tag=t,
                scalar_value=score,
                global_step=phase_index,
            )
            metrics_string += f",{score}"
            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                # torch.cuda.reset_max_memory_cached()
        metrics_string += "\n"
        csv.write(metrics_string)
        csv.close()

    def add_audio(self, tag=None, audio=None, phase_index=0, sample_rate=None, walltime=None):
        if audio is None:
            return
        tag = tag or Tabs.AUDIO
        self.writer.add_audio(
            tag, audio,
            # global_step=phase_index,
            # sample_rate=sample_rate,
            # walltime=walltime
        )

    def add_histogram(self, values=None, phase_index=0, name=None):
        if values is None:
            return
        # name = name or 'Default'
        tag = f"{Tabs.HIST}/{name}" if name else Tabs.HIST
        self.writer.add_histogram(
            tag=tag, values=values, global_step=phase_index
        )

    def add_embedding(self, embedding=None, tag=None, phase_index=0):
        if embedding is None or tag is None:
            return
        tag = f"{Tabs.EMBED}/{tag}" if tag else Tabs.EMBED
        self.writer.add_embedding(
            embedding,
            # metadata=None,
            # label_img=None,
            global_step=phase_index,
            tag=tag,
            metadata_header=None
        )

    def add_graph(self, model=None, inputs=None):
        if model is None or inputs is None:
            return
        self.writer.add_graph(
            model, input_to_model=inputs, verbose=False, use_strict_trace=True
        )

    def add_images(self, tag=None, images=None):
        if images is None:
            return
        tag = tag or Tabs.IMAGE
        self.writer.add_images(tag, images)

    def on_update(
        self,
        iteration=0,
        loss=None,
        log_frequency=None,
        batch_time=None,
        max_iteration=None
    ):
        """
        Call after every parameter's update if tensorboard logger is enabled.
        Logs the scalars like training loss, learning rate, average training
        iteration time, ETA, gpu memory used, peak gpu memory used.
        """
        batch_time = batch_time or []
        if (
            log_frequency is not None
            and iteration % log_frequency == 0
            or (iteration <= 100 and iteration % 5 == 0)
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            self.writer.add_scalar(
                tag=f"{Tabs.TRAIN}/Loss",
                scalar_value=round(loss.data.cpu().item(), 5),
                global_step=iteration
            )

            # TODO: lr saving
            # self.tb_writer.add_scalar(
            #     tag="Training/Learning_rate",
            #     scalar_value= ...,
            #     global_step=iteration,
            # )

            # Batch processing time
            if len(batch_time) > 0:
                batch_times = batch_time
            else:
                batch_times = [0]

            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            self.writer.add_scalar(
                tag=f"{Tabs.SPEED}/Batch_processing_time_ms",
                scalar_value=int(1000.0 * batch_time_avg_s),
                global_step=iteration,
            )

            # ETA
            if max_iteration is not None:
                avg_time = sum(batch_times) / len(batch_times)
                eta_secs = avg_time * (max_iteration - iteration)
                self.writer.add_scalar(
                    tag=f"{Tabs.SPEED}/ETA_hours",
                    scalar_value=eta_secs / 3600.0,
                    global_step=iteration,
                )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                self.writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved() / BYTE_TO_MiB,
                    global_step=iteration,
                )

    def __del__(self):
        self.writer.close()


__all__ = ['tensorboard_logger_factory', 'LoggerTensorboard']
