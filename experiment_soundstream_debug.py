import argparse

from datetime import datetime
from os import path as osp

from pnc.config import OmegaConf as OMG
from pnc.OmniLog import LoggerUltimate
from pnc.soundstream.soundstream import SoundStream
from pnc.soundstream.trainer import SoundStreamTrainer
from pnc.soundstream.debug_dataset import make_placeholder_dataset

parser = argparse.ArgumentParser()

timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')

parser.add_argument(
    "-c",
    "--config",
    default="./config_default.yaml",
    help="config path"
)

parser.add_argument(
    "-n",
    "--run-name",
    dest="experiment.name",
    default=f"{timestamp}",
    help=f"experiment/run name (default: {timestamp})"
)

parser.add_argument(
    "-o",
    "--run-output",
    dest="experiment.path_output",
    default="../results",
    help="experiment/run output directory root"
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="run in debug mode"
)

parser.add_argument(
    "-d",
    "--device",
    dest='env.device',
    default=0,
    help="select device (CPU | 0 | 1 | 2 | ...)"
)

# args = parser.parse_args()
cfg = OMG.from_cli(parser)  # omg.load(args.config)

run_folder = osp.join(cfg.experiment.path_output, cfg.experiment.name)

if cfg.debug:
    cfg.trainer.num_train_steps = 10
    # folder = cfg.trainer.dataset_folder
    folder = './debug_dataset/'
    cfg.trainer.dataset_folder = folder
    make_placeholder_dataset(folder)

soundstream = SoundStream(
    codebook_size=cfg.model.codebook_size,
    rq_num_quantizers=cfg.model.rq_num_quantizers,
)

print(cfg)

logger = LoggerUltimate(cfg, online_logger='tensorboard')

trainer = SoundStreamTrainer(
    soundstream,
    folder=cfg.trainer.dataset_folder,
    batch_size=cfg.trainer.batch_size,
    grad_accum_every=cfg.trainer.grad_accum_every,
    data_max_length=cfg.trainer.data_max_length,
    save_results_every=cfg.trainer.save_results_every,
    save_model_every=cfg.trainer.save_model_every,
    num_train_steps=cfg.trainer.num_train_steps,
    valid_frac=cfg.trainer.valid_frac,
    run_folder=run_folder,
    logger=logger
    # accelerate_kwargs={'log_with': logger}
).to(cfg.env.device)

trainer.train()
logger.shutdown_logging()
exit(0)
