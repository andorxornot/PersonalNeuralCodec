from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf as OmegaConfBase

from ..logs import LoggerBase, MessageType


class OmegaConf(OmegaConfBase):
    """Extended OmegaConf class that support update from argparse.ArgumentParser

    See https://github.com/omry/omegaconf/issues/569
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def unflatten(
            source: Dict[str, Any], sep: str = '.', skip_nones: bool = True,
            logs: LoggerBase = None
    ) -> Optional[Dict[str, Any]]:
        """Unflatten a dictionary by key separator

        Produce a hierarchical dictionary by separators in keys.

        Args:
            source (Dict[str, Any]): a dictionary containing flat config values
            sep (str): separator to split the flat dictionary keys by
            skip_nones (bool): skip branches with None values
            logs (LoggerBase): logger to send messages to

        Returns:
            Dict[str, Any]: hierarchical config dictionary

        Examples:
            > OmegaConf.unflatten({"debug": False, "trainer.batch_size": 8})
            {"debug": False, "trainer": {"batch_size": 8}}
        """
        nested: Dict[str, Any] = defaultdict(dict)
        expand = lambda k, v: {k[0]: v} if len(k) == 1 else {k[0]: expand(k[1:], v)}

        def nest(n: Dict[str, Any], e: Any):
            for k, v in e.items():
                if isinstance(v, dict):
                    n[k] = n.get(k, {})
                    nest(n[k], e[k])
                else:
                    try:
                        n[k] = v
                    except TypeError:
                        if isinstance(logs, LoggerBase):
                            logs.log(f"skipping branch { {k: e[k]} }...",
                                     MessageType.WARNING)

        for key, value in source.items():
            if value is None and skip_nones:
                continue
            key_split = key.split(sep)
            expansion = expand(key_split, value)
            nest(nested, expansion)

        return dict(nested)

    @staticmethod
    def from_cli(
            args: Optional[Union[ArgumentParser, Namespace, List[str]]] = None,
            sep: str = '.', skip_nones: bool = True,
            logs: Optional[LoggerBase] = None
    ) -> Union[DictConfig, ListConfig]:
        """from_argparse Static method to convert argparse arguments into OmegaConf DictConfig objects

        We parse the command line arguments and separate the user provided values and the default values.
        This is useful for merging with a config file.

        Args:
            args (Optional[Union[ArgumentParser, Namespace, List[str]]]): arguments
            sep (str): separator to split the flat dictionary keys by
            skip_nones (bool): skip branches with None values
            logs (LoggerBase): logger to send messages to
        Returns:
            Union[DictConfig, ListConfig]: depends on OmegaConf.merge

        Examples:
            > from argparse import ArgumentParser
            > from pnc.config import OmegaConf
            > parser = ArgumentParser("This is just an example")
            > parser.add_argument("--config", default="config_default.yaml")
            > parser.add_argument("--batch-size", dest="trainer.batch_size", type=int, default=4)
            > print(OmegaConf.from_cli(parser))
            {'debug': False, ... 'trainer': {'batch_size': 4}}
        """
        # Default behaviour if args is just a list of sys.argv (as superclass)
        if not isinstance(args, (ArgumentParser, Namespace)):
            return super(__class__, __class__).from_cli(args)

        if isinstance(args, ArgumentParser):
            args = args.parse_args()
        args = deepcopy(args)
        args = vars(args)

        try:
            config_path = args['config']
            del args['config']
            config_base = super(__class__, __class__).load(config_path)
        except (IsADirectoryError, KeyError) as ex:
            if isinstance(ex, IsADirectoryError) and isinstance(logs, LoggerBase):
                logs.log(f"path '{config_path}' must be a valid config file",
                         MessageType.ERROR)
            config_base = super(__class__, __class__).create({})
        print(config_base)
        config_args = super(__class__, __class__).create(__class__.unflatten(
            source=args, sep=sep, skip_nones=skip_nones, logs=logs
        ))
        print(config_args)

        return super(__class__, __class__).merge(config_base, config_args)
