import logging
import subprocess

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from enum import Enum
from os import environ, makedirs, path as osp
from sys import stdout as STDOUT

import torch

from ..utils import pytorch_worker_info


class MessageType(Enum):
    NONE = 0  # NOTSET
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FATAL = 5  # CRITICAL


FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"


class LoggerOnline(metaclass=ABCMeta):
    """
    Online logger interface (abstract class) with certain methods to implement.
    """
    @abstractmethod
    def add_audio(self):
        ...

    @abstractmethod
    def add_embedding(self):
        ...

    @abstractmethod
    def add_graph(self):
        ...

    @abstractmethod
    def add_histogram(self):
        ...

    @abstractmethod
    def add_images(self):
        ...

    @abstractmethod
    def log_metrics(self):
        ...

    @abstractmethod
    def log_scalars(self):
        ...

    @abstractmethod
    def on_update(self):
        ...

    @abstractmethod
    def set_run(self):
        ...

    @contextmanager
    @abstractmethod
    def toggle_run(self):
        ...


class LoggerBase:
    """
    Base logger class.
    """
    def __init__(self, name: str, path_target: str = None):
        """
        Initializer of the custom base logger class.
        Args:
            name (str): the logger name
            path_target (str): experiment logs base output directory
        """
        rank, world_size, _, _ = pytorch_worker_info()
        if world_size and world_size > 1:
            self.use_ddp = int(environ.get("USE_DDP", "1"))
            if not self.use_ddp:
                self.multi_gpu = True
        else:
            self.use_ddp = False
            self.multi_gpu = False

        log_filename = None

        if path_target and not rank:
            makedirs(path_target, exist_ok=True)
        if not self.use_ddp or not rank:
            log_filename = f"{path_target}/log_{rank or 0}.txt"

        if not self.use_ddp or not rank:
            self.logger_stdout, self.logger_file = self._init_loggers(
                name, log_filename
            )
        else:
            self.logger_stdout, self.logger_file = None, None
        self.output_dir = path_target

    def _init_loggers(self, name: str, log_filename: str, log_format: str = None):
        """
        Initializer for two loggers: log to stdout and log to file.

        Args:
            name (str): the logger name
            log_filename (str): logger file name (for file logger)
            log_format (str): format string for both loggers

        Returns:
            Logger: stdout logger
            Union[Logger, None]: file logger or None if no filename provided

        """
        logger_stdout = logging.getLogger(f"{name}_stdout")
        logger_stdout.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(log_format or FORMAT)

        # Set up the console handler
        handler_stdout = logging.StreamHandler(STDOUT)
        handler_stdout.setLevel(logging.INFO)
        handler_stdout.setFormatter(formatter)

        logger_stdout.addHandler(handler_stdout)
        logger_stdout.propagate = False

        # We log to file as well if user wants
        logger_file = None
        if log_filename:
            logger_file = logging.getLogger(f"{name}_filewriter")
            logger_file.setLevel(logging.INFO)
            handler_file = logging.FileHandler(log_filename)
            handler_file.setLevel(logging.INFO)
            handler_file.setFormatter(formatter)
            logger_file.addHandler(handler_file)
            logger_file.propagate = False
        return logger_stdout, logger_file

    def log(
        self,
        message: str,
        mtype: MessageType = None,
        main_rank_only: bool = False,
        to_stdout: bool = True,
        to_file: bool = True
    ):
        """
        Logs message to stdout/file.

        Args:
            message (str): message to log
            mtype (MessageType): message type (DEBUG, INFO, WARNING, ERROR, FATAL)
            main_rank_only (bool): log the process with rank 0 only
            to_stdout (bool): whether write to stdout (default: True)
            to_file (bool): whether write to file (default: True)

        Returns:
            None:

        """
        rank, _, _, _ = pytorch_worker_info()  # TODO: avoid redundant calls
        if (self.use_ddp or main_rank_only) and rank:
            return

        if mtype == MessageType.DEBUG or mtype == MessageType.DEBUG.value:
            if to_stdout:
                self.logger_stdout.debug(message)
            if to_file:
                self.logger_file.debug(message)
        elif mtype == MessageType.WARNING or mtype == MessageType.WARNING.value:
            if to_stdout:
                self.logger_stdout.warning(message)
            if to_file:
                self.logger_file.warning(message)
        elif mtype == MessageType.ERROR or mtype == MessageType.ERROR.value:
            if to_stdout:
                self.logger_stdout.error(message)
            if to_file:
                self.logger_file.error(message)
        elif mtype == MessageType.FATAL or mtype == MessageType.FATAL.value:
            if to_stdout:
                self.logger_stdout.critical(message)
            if to_file:
                self.logger_file.critical(message)
        else:
            if to_stdout:
                self.logger_stdout.info(message)
            if to_file:
                self.logger_file.info(message)

    def shutdown_logging(self):
        """
        After training is done, we ensure to shut down all the logger streams.
        """
        # TODO: make static or use self
        logging.info("Shutting down loggers...")
        handlers = logging.root.handlers  # Why not self.logger*s ?
        for handler in handlers:
            handler.close()

    def log_gpu_stats(self):
        """
        Log nvidia-smi snapshot. Useful to capture the configuration of gpus.
        """
        try:
            self.log(
                subprocess.check_output(["nvidia-smi"]).decode("utf-8"),
                MessageType.INFO,
                main_rank_only=True,
            )
        except FileNotFoundError:
            self.log(
                "Failed to find the 'nvidia-smi' executable for printing GPU stats",
                MessageType.ERROR,
                main_rank_only=True,
            )
        except subprocess.CalledProcessError as e:
            self.log(
                f"nvidia-smi returned non zero error code: {e.returncode}",
                MessageType.ERROR,
                main_rank_only=True,
            )

    def print_gpu_memory_usage(self):
        """
        Parse the nvidia-smi output and extract the memory used stats.
        Not recommended to use.
        """
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        all_values, count, out_dict = [], 0, {}
        for item in out_list:
            if " MiB" in item:
                out_dict[f"GPU {count}"] = item.strip()
                all_values.append(int(item.split(" ")[0]))
                count += 1
        self.log(
            f"Memory usage stats:\n"
            f"Per GPU mem used: {out_dict}\n"
            f"nMax memory used: {max(all_values)}",
            MessageType.INFO,
            main_rank_only=True,
        )

    def _prepare_tree_for_saving(self, name: str, subdir: str = None):
        """
        Prepare full path for saving files.

        Args:
            name (str): file name
            subdir (str): subdirectory

        Returns:
            None:

        """
        rank, _, _, _ = pytorch_worker_info()  # TODO: avoid redundant calls
        if self.use_ddp and rank:
            return None

        if subdir is not None:
            out_folder = osp.join(self.output_dir, subdir)
            out_file = osp.join(out_folder, name)
            makedirs(out_folder, exist_ok=True)
        else:
            out_file = osp.join(self.output_dir, name)

        return out_file

    def save_custom_txt(
        self, content: str = "", name: str = "name.txt", subdir: str = None
    ):
        """
        Allow to save custom txt file.

        Args:
            content (str): message
            name (str): file name
            subdir (str): subdirectory

        Returns:
            None:

        """
        out_file = self._prepare_tree_for_saving(name, subdir)
        if out_file is None:
            return
        with open(out_file, "w") as f:
            f.write(content)

    def save_custom_torch(self, content: str, name: str, subdir: str = None):
        """
        Allow to save custom torch file.

        Args:
            content (str): message
            name (str): file name
            subdir (str): subdirectory

        Returns:
            None:

        """
        out_file = self._prepare_tree_for_saving(name, subdir)
        if out_file is None:
            return
        torch.save(content, out_file)


__all__ = ['LoggerBase', 'LoggerOnline', 'MessageType']
