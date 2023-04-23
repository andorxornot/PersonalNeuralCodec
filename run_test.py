import os
import argparse
from omegaconf import OmegaConf as omg
from external.lucid_soundstream.soundstream import SoundStream 
from external.lucid_soundstream.trainer import SoundStreamTrainer 
from external.lucid_soundstream.debug_dataset import make_placeholder_dataset


parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    default='./base_config.yaml',
    help="confir paht",
)


args = parser.parse_args()
conf = omg.load(args.config)



if conf.debug:
     conf.trainer.num_train_steps = 10
     folder = conf.trainer.dataset_folder
     folder = './debug_dataset/'
     conf.trainer.dataset_folder = folder
     make_placeholder_dataset(folder)

soundstream = SoundStream(
    codebook_size = conf.model.codebook_size,
    rq_num_quantizers = conf.model.rq_num_quantizers,
)

print(conf)

trainer = SoundStreamTrainer(
    soundstream,
    folder = conf.trainer.dataset_folder,
    batch_size = conf.trainer.batch_size,
    grad_accum_every = conf.trainer.grad_accum_every,       
    data_max_length =  conf.trainer.data_max_length,
    save_results_every =  conf.trainer.save_results_every,
    save_model_every =  conf.trainer.save_model_every,
    num_train_steps =  conf.trainer.num_train_steps,
    valid_frac = conf.trainer.valid_frac,
).to(conf.env.device)

trainer.train()