import os
import sys
import argparse
import time
import datetime

# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'AcademiCodec', 'SoundStream_24k_240d'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from pathlib import Path
import torchaudio
import itertools

from external.AcademiCodec.SoundStream_24k_240d.models import MultiPeriodDiscriminator, MultiScaleDiscriminator
from external.AcademiCodec.SoundStream_24k_240d.msstftd import MultiScaleSTFTDiscriminator
from external.AcademiCodec.SoundStream_24k_240d.net3 import SoundStream
from external.AcademiCodec.SoundStream_24k_240d.dataset import NSynthDataset
from external.AcademiCodec.SoundStream_24k_240d.utils import seed_everything, Logger
from external.AcademiCodec.SoundStream_24k_240d.loss import loss_g, loss_dis, criterion_g, criterion_d

from pnc.config import OmegaConf as OMG
from pnc.logger_united import LoggerUnited

def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()

def check_clipping2(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Total model size is：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def save_audio(wav: torch.Tensor, path: str, sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def omegaconf_to_namespace(omegaconf_obj):
    namespace = argparse.Namespace()
    for key, value in OMG.to_container(omegaconf_obj, resolve=True).items():
        setattr(namespace, key, value)
    return namespace

def main():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default="./exp_configs/experiment_soundstream_2_debug.yaml", help="config path")
    parser.add_argument("-n", "--run-name", dest="experiment.name", default="{}".format(timestamp), help="experiment/run name (default: {})".format(timestamp))
    parser.add_argument("-o", "--run-output", dest="experiment.path_output", default="../results", help="experiment/run output directory root")
    parser.add_argument("-d", "--device", dest='env.device', default=0, type=int, help="select device (CPU | 0 | 1 | 2 | ...)" )
    parser.add_argument("--debug", action="store_true", help="run in debug mode")

    cfg = OMG.from_cli(parser) 
    logger = LoggerUnited(cfg, online_logger='tensorboard')

    # logs
    if cfg.env.resume:
        cfg.env.PATH = cfg.env.resume_path # direcly use the old model path
    else:
        cfg.env.PATH = os.path.join(cfg.env.PATH, timestamp)
    os.makedirs(cfg.env.PATH, exist_ok=True)
    
    if cfg.env.seed is not None or cfg.env.cudnn_deterministic:
        seed_everything(cfg.env.seed, cfg.env.cudnn_deterministic)
 

    soundstream = SoundStream(n_filters=32, D=512, ratios=[6, 5, 4, 2]) # 240 times lower picking
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    print('soundstream ', soundstream)
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)
    getModelSize(soundstream)
    getModelSize(msd)
    getModelSize(mpd)
    getModelSize(stft_disc)
    
    device = torch.device(cfg.env.device)
    args = omegaconf_to_namespace(cfg.env)
    args.device = device

    soundstream.to(device)
    stft_disc.to(device)
    msd.to(device)
    mpd.to(device)
    train_dataset = NSynthDataset(audio_dir=cfg.env.train_data_path)
    valid_dataset = NSynthDataset(audio_dir=cfg.env.valid_data_path)
    cfg.env.sr = train_dataset.sr
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.env.BATCH_SIZE, num_workers=8, sampler=None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.env.BATCH_SIZE, num_workers=8, sampler=None)
    optimizer_g = torch.optim.AdamW(soundstream.parameters(), lr=3e-4, betas=(0.5, 0.9))
    lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    optimizer_d = torch.optim.AdamW(itertools.chain(stft_disc.parameters(),msd.parameters(),mpd.parameters()),
                                     lr=3e-4, betas=(0.5, 0.9))
    lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)
    
    if cfg.env.resume:
        latest_info = torch.load(cfg.env.resume_path+'/latest.pth')
        cfg.env.st_epoch = latest_info['epoch']
        soundstream.load_state_dict(latest_info['soundstream'])
        stft_disc.load_state_dict(latest_info['stft_disc'])
        mpd.load_state_dict(latest_info['mpd'])
        msd.load_state_dict(latest_info['msd'])
        optimizer_g.load_state_dict(latest_info['optimizer_g'])
        lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])
        optimizer_d.load_state_dict(latest_info['optimizer_d'])
        lr_scheduler_d.load_state_dict(latest_info['lr_scheduler_d'])

    # train
    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = 0

    for epoch in range(cfg.env.st_epoch, cfg.env.N_EPOCHS+1):
        soundstream.train()
        stft_disc.train()
        msd.train()
        mpd.train()
        train_loss_d = 0.0
        train_adv_g_loss = 0.0
        train_feat_loss = 0.0
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_commit_loss = 0.0
        k_iter = 0 

        for x in tqdm(train_loader):
            x = x.to(device)
            k_iter += 1
            global_step += 1 # record the global step
            for optimizer_idx in [0,1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer = soundstream(x_wav)
                if optimizer_idx==0:
                    # update generator

                    y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                    y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x_wav.contiguous(), G_x.contiguous())
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x_wav.contiguous(), G_x.contiguous())
                    total_loss_g, rec_loss, adv_g_loss, feat_loss, d_weight = loss_g(
                        commit_loss, x_wav, G_x, fmap_r, fmap_gen, 
                        y_disc_r, y_disc_gen, global_step, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g,
                        fmap_f_r, fmap_f_g, fmap_s_r, fmap_s_g,
                        last_layer=last_layer, is_training=True, args=args
                    )
                    train_commit_loss += commit_loss 
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()

                    total_loss_g.backward()
                    if global_step > 0 and (global_step + 1) % cfg.env.grad_accum_every == 0:
                        # torch.nn.utils.clip_grad_norm_(soundstream.parameters(), 1.0)

                        optimizer_g.step()
                        optimizer_g.zero_grad()
                else:
                    # update discriminator
                    y_disc_r_det, fmap_r_det = stft_disc(x.detach())
                    y_disc_gen_det, fmap_gen_det = stft_disc(G_x.detach())

                    # MPD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x.detach(), G_x.detach())
                    #MSD
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x.detach(), G_x.detach())

                    loss_d = loss_dis(y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det, 
                                      y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, 
                                      y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g, global_step, args)
                    train_loss_d += loss_d.item()

                    loss_d.backward()
                    if global_step > 0 and (global_step + 1) % cfg.env.grad_accum_every == 0:
                        #torch.nn.utils.clip_grad_norm_(list(stft_disc.parameters()) + list(msd.parameters()) + list(mpd.parameters()), 1.0)
       
                        optimizer_d.step()
                        optimizer_d.zero_grad()

            logger.log({
                'epoch': epoch,
                'iter': k_iter,
                'total_loss_g': total_loss_g.item(),
                'adv_g_loss': adv_g_loss.item(),
                'feat_loss': feat_loss.item(),
                'rec_loss': rec_loss.item(),
                'commit_loss': commit_loss.item(),
                'loss_d': loss_d.item(),
                'd_weight': d_weight.item()
            }, global_step=global_step)

            if k_iter % cfg.env.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, loss_d:{:.4f}>, d_weight: {:.4f}'.format(epoch, 
                    k_iter, total_loss_g.item(), adv_g_loss.item(), feat_loss.item(), rec_loss.item(), commit_loss.item(), loss_d.item(), d_weight.item())
                print(message)                

            if global_step % cfg.env.save_results_steps == 0:
                #test
                with torch.no_grad():
                    rescale = True # Automatically rescale the output to avoid clipping
                    x = valid_dataset[0]
                    x_wav = get_input(x)
                    #compressing
                    wav = x_wav.unsqueeze(1).to(device)
                    print('wav ', wav.shape)
                    compressed = soundstream.encode(wav)
                    # print('compressed ', compressed.shape)
                    print('finish compressing')
                    out = soundstream.decode(compressed)
                    out = out.detach().cpu().squeeze(0)
                    check_clipping2(out, rescale)
                    save_audio(x_wav, os.path.join(cfg.env.PATH, "in.wav"), cfg.env.sr, rescale=rescale)
                    save_audio(out, os.path.join(cfg.env.PATH, 'out_{:010d}.wav'.format(global_step)), cfg.env.sr, rescale=rescale)

                    logger.add_audio(tag='Audio/orig',audio=x_wav, phase_index=global_step, sample_rate=cfg.env.sr)
                    logger.add_audio(tag='Audio/recon',audio=out, phase_index=global_step, sample_rate=cfg.env.sr)

                    print('finish decompressing')

        lr_scheduler_g.step()
        lr_scheduler_d.step()
        message = '<epoch:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}>'.format(epoch,
                   train_loss_g/len(train_loader),train_rec_loss/len(train_loader), 
                   train_adv_g_loss/len(train_loader),train_feat_loss/len(train_loader),
                   train_commit_loss/len(train_loader))
        print(message)                

        logger.log({
            'epoch': epoch,
            'total_loss_g_train': train_loss_g/len(train_loader),
            'recon_loss_train': train_rec_loss/len(train_loader),
            'adversarial_loss_train': train_adv_g_loss/len(train_loader),
            'feature_loss_train': train_feat_loss/len(train_loader),
            'commit_loss_train': train_commit_loss/len(train_loader)
        }, global_step=global_step)

        # val
        with torch.no_grad():
            soundstream.eval()
            stft_disc.eval()
            mpd.eval()
            msd.eval()
            valid_loss_d = 0.0
            valid_loss_g = 0.0
            valid_commit_loss = 0.0
            valid_adv_g_loss = 0.0
            valid_feat_loss = 0.0
            valid_rec_loss = 0.0

            for x in tqdm(valid_loader):
                x = x.to(device)
                for optimizer_idx in [0,1]: 
                    x_wav = get_input(x)
                    G_x, commit_loss, _ = soundstream(x_wav)
                    if optimizer_idx == 0:
                        valid_commit_loss += commit_loss
                        y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                        y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x_wav.contiguous(), G_x.contiguous())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x_wav.contiguous(), G_x.contiguous())

                        total_loss_g, adv_g_loss, feat_loss, rec_loss = criterion_g(
                            commit_loss, x_wav, G_x, fmap_r, fmap_gen, 
                            y_disc_r, y_disc_gen, y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g,
                            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g, args=args)   
                        valid_loss_g += total_loss_g.item()
                        valid_adv_g_loss += adv_g_loss.item()
                        valid_feat_loss += feat_loss.item()
                        valid_rec_loss += rec_loss.item()
                    else:
                        y_disc_r_det, fmap_r_det = stft_disc(x_wav.contiguous().detach())
                        y_disc_gen_det, fmap_gen_det = stft_disc(G_x.contiguous().detach())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x_wav.contiguous().detach(), G_x.contiguous().detach())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x_wav.contiguous().detach(), G_x.contiguous().detach())
                        loss_d = criterion_d(y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det,
                                            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g,
                                            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g)
                        valid_loss_d += loss_d.item()
            
            best_model = soundstream.state_dict().copy()
            latest_model_soundstream = soundstream.state_dict().copy()
            latest_model_dis = stft_disc.state_dict().copy()
            latest_mpd = mpd.state_dict().copy()
            latest_msd = msd.state_dict().copy()
            if valid_rec_loss < best_val_loss:
                best_val_loss = valid_rec_loss
                best_val_epoch = epoch
            
            if epoch % cfg.env.save_model_epoch == 0:
                torch.save(best_model, cfg.env.PATH + '/best_'+str(epoch)+'.pth')

            latest_save = {}
            latest_save['soundstream'] = latest_model_soundstream
            latest_save['stft_disc'] = latest_model_dis
            latest_save['mpd']  = latest_mpd
            latest_save['msd'] = latest_msd
            latest_save['epoch'] = epoch
            latest_save['optimizer_g'] = optimizer_g.state_dict()
            latest_save['optimizer_d'] = optimizer_d.state_dict()
            latest_save['lr_scheduler_g'] = lr_scheduler_g.state_dict()
            latest_save['lr_scheduler_d'] = lr_scheduler_d.state_dict()
            torch.save(latest_save, cfg.env.PATH +'/latest.pth')
            
            message = '<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, best_epoch:{:d}>'.format(epoch,
                   valid_loss_g/len(valid_loader), valid_rec_loss/len(valid_loader), 
                   valid_adv_g_loss/len(valid_loader), valid_feat_loss/len(valid_loader),
                   valid_commit_loss/len(valid_loader),valid_loss_d/len(valid_loader), best_val_epoch)
            print(message)
            
            logger.log({
                'epoch': epoch,
                'total_loss_g_valid': valid_loss_g/len(valid_loader),
                'recon_loss_valid': valid_rec_loss/len(valid_loader),
                'adversarial_loss_valid': valid_adv_g_loss/len(valid_loader),
                'feature_loss_valid': valid_feat_loss/len(valid_loader),
                'commit_loss_valid': valid_commit_loss/len(valid_loader),
                'valid_loss_d': valid_loss_d/len(valid_loader),
                'best_epoch': best_val_epoch
            }, global_step=global_step)


    logger.shutdown_logging()

if __name__ == '__main__':
    main()           
        