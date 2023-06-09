import logging

from contextlib import contextmanager
from dataclasses import dataclass
from os import path as osp

import torch

from .base import LoggerOnline
from .checkpoint import get_checkpoint_folder
from .io import is_file_empty, makedirs


def get_paths(cfg, name, log_dir):
    """
    Get the output directory where the tensorboard events will be written.
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        Tensorboard as well as log directory, logging frequency etc.
    Returns:
        path_tensorboard (str): tensorboard output directory path
    """

    path_logs = osp.join(cfg.experiment.path_output, cfg.experiment.name, log_dir)
    print(f"{name} path = {path_logs}") 
    makedirs(path_logs, exist_ok=True)
    return path_logs

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

def wandb_available():
    """
    Check whether tensorboard is available or not.
    Returns:
        tb_available (bool): based on tensorboard imports, returns whether
                             TensorBoard is available or not.
    """
    try:
        import wandb  
        
        wandb_available = True
    except ImportError:
        logging.warning("Wandb is not available")
        wandb_available = False
    return wandb_available


def logger_factory(cfg):
    """
    Construct the Tensorboard logger for visualization from the specified config
    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well as log directory, logging frequency etc.
    Returns:
        TensorboardHook (function): the tensorboard logger constructed
    """

    # Get the tensorboard directory and check tensorboard is installed
    path_tensorboard  = get_paths(cfg, 'Tensorboard', 'tb')
    path_csv  = get_paths(cfg, 'CSV', 'csv')
    flush_secs = cfg.logging.TENSORBOARD.FLUSH_EVERY_N_MIN * 60
    return MainLogger(
        exp_name = cfg.experiment.name,
        path_tb=path_tensorboard,
        path_csv=path_csv,
        use_wdb = cfg.logging.use_wandb,
        flush_secs=flush_secs)


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


class MainLogger(LoggerOnline):
    """
    Tensorboard logger
    """
    def __init__(
        self,
        exp_name,
        path_tb,
        path_csv,
        use_wdb,
        flush_secs,
        log_params=False,
        log_params_frequency=-1
    ):
        """
        The constructor method of LoggerTensorboard.
        Args:
            path_tb (str): Tensorboard base directory
            path_csv (str): directory path to store metrics as csv files
        """


        if wandb_available() and use_wdb:
            import wandb 
            wandb.tensorboard.patch(root_logdir=path_tb)
            wandb.init(project=exp_name, dir=path_tb, sync_tensorboard=True)

        if not tensorboard_available():
            raise RuntimeError(
                "tensorboard not installed, cannot use 'LoggerTensorboard'"
            )
        from torch.utils.tensorboard import SummaryWriter

        self.tb_class = SummaryWriter
        logging.info("Setting up Tensorboard Logger...") 
        self.path_tb = path_tb
        self.path_csv = path_csv
        self.cur_run = None
        self._runs_stack = []
        self.flush_secs = flush_secs
        self.tb_writer = self.tb_class(log_dir=path_tb, flush_secs=flush_secs)
        self.use_wdb = use_wdb
     

    def set_run(self, run_name=None):
        log_dir = self.path_tb
        if run_name is not None:
            log_dir += f"/{run_name}"
        self.tb_writer.close()
        self.tb_writer = self.tb_class(log_dir=log_dir, flush_secs=self.flush_secs)

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
        csv = open(csv_name, "a")
        if is_file_empty(csv_name):
            csv.write(f"phase_index,{','.join(scalars.keys())}\n")
        # Log train/test accuracy
        metrics_string = f"{phase_index}"
        self.tb_writer.add_scalars(
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
        csv = open(csv_name, "a")
        if is_file_empty(csv_name):
            csv.write(f"phase_idx,process,{','.join(metrics.keys())}\n")
        # Log train/test accuracy
        metrics_string = f"{phase_index},{rank}"

        for metric_name, score in metrics.items():
            t = f"{tab}/Process {rank}/{metric_name}"
            self.tb_writer.add_scalar(
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
        self.tb_writer.add_audio(
            tag, audio,
            global_step=phase_index,
            sample_rate=sample_rate,
        )


    def add_histogram(self, values=None, phase_index=0, name=None):
        if values is None:
            return
        # name = name or 'Default'
        tag = f"{Tabs.HIST}/{name}" if name else Tabs.HIST
        self.tb_writer.add_histogram(
            tag=tag, values=values, global_step=phase_index
        )

    def add_embedding(self, embedding=None, tag=None, phase_index=0):
        if embedding is None or tag is None:
            return
        tag = f"{Tabs.EMBED}/{tag}" if tag else Tabs.EMBED
        self.tb_writer.add_embedding(
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
        self.tb_writer.add_graph(
            model, input_to_model=inputs, verbose=False, use_strict_trace=True
        )

    def add_images(self, tag=None, images=None):
        if images is None:
            return
        tag = tag or Tabs.IMAGE
        self.tb_writer.add_images(tag, images)

    def add_audio(self,  tag, snd_tensor, global_step=None, sample_rate=44100):
        tag = tag or  Tabs.AUDIO
        self.tb_writer.add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)

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
            self.tb_writer.add_scalar(
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
            self.tb_writer.add_scalar(
                tag=f"{Tabs.SPEED}/Batch_processing_time_ms",
                scalar_value=int(1000.0 * batch_time_avg_s),
                global_step=iteration,
            )

            # ETA
            if max_iteration is not None:
                avg_time = sum(batch_times) / len(batch_times)
                eta_secs = avg_time * (max_iteration - iteration)
                self.tb_writer.add_scalar(
                    tag=f"{Tabs.SPEED}/ETA_hours",
                    scalar_value=eta_secs / 3600.0,
                    global_step=iteration,
                )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.tb_writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.tb_writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                self.tb_writer.add_scalar(
                    tag=f"{Tabs.MEMORY}/Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved() / BYTE_TO_MiB,
                    global_step=iteration,
                )

    def __del__(self):
        self.tb_writer.close()
        if self.use_wdb:
            wandb.finish()


__all__ = ['logger_factory', 'MainLogger']

