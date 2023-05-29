# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line for audio compression."""
import argparse
from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
from collections import OrderedDict
import librosa
import glob
import soundfile as sf
import librosa
import os
import glob
import argparse
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
import scipy.signal as signal
from pystoi import stoi
import numpy as np


# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'AcademiCodec', 'Encodec_16k_320'))

from external.AcademiCodec.Encodec_16k_320.net3 import SoundStream



SUFFIX = '.ecdc'
def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        '--input', default=Path('/home/k4/Projects/PWC/PersonalNeuralCodec/data/music_samples'), type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        '--output', default=Path('//home/k4/Projects/PWC/PersonalNeuralCodec/data/music_samples_16'),  type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    
    parser.add_argument('--resume_path', type=str, default='/home/k4/Projects/PWC/AcademiCodec/encodec_16k_320d.pth',  help='resume_path')
    parser.add_argument('--samplerate', type=int, default=16000)
    parser.add_argument('--target_bw', type=float, default=12)
    parser.add_argument('--ratios', type=int, nargs='+', default=[8, 5, 4, 2], help='List of ratios')
	
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    return parser


def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


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


def test_one(wav_input, wav_output, rescale, args, soundstream):
    #compressing
    samplerate = args.samplerate
    wav, sr = sf.read(wav_input)
    wav = librosa.resample(wav, orig_sr= sr, target_sr= samplerate)

    if not os.path.exists(os.path.dirname(wav_output)):
        os.makedirs(os.path.dirname(wav_output))

    #sf.write(outpath, wav, samplerate)

    wav = torch.from_numpy(wav).float().unsqueeze(0)

    wav = wav.unsqueeze(1).cuda()
    #print('wav ', wav.shape)
    compressed = soundstream.encode(wav, target_bw=args.target_bw)
    #print('compressed ', compressed) # (n_q, B, len)
    
    # assert 1==2
    #print(wav_input)
    #print('finish compressing')
    out = soundstream.decode(compressed)
    # print('out ', out.shape)
    # assert 1==2
    out = out.detach().cpu().squeeze(0)
    check_clipping2(out, rescale)

    save_audio(out, wav_output, samplerate, rescale=rescale)
    print('finish decompressing')
    #assert 1==2
 

def cal_pesq(ref_dir, deg_dir, samplerate):
    input_files = glob.glob(f"{deg_dir}/**/*.wav", recursive=True)
    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.relpath(deg_wav, deg_dir) )
        ref_rate, ref = wavfile.read(ref_wav)
        deg_rate, deg = wavfile.read(deg_wav)
        if ref_rate != samplerate:
            ref = signal.resample(ref, samplerate)
        if deg_rate != samplerate:
            deg = signal.resample(deg, samplerate)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        nb_pesq_scores += pesq(samplerate, ref, deg, 'nb')
        wb_pesq_scores += pesq(samplerate, ref, deg, 'wb')

    return  nb_pesq_scores/len(input_files), wb_pesq_scores/len(input_files)


def calculate_stoi(ref_dir, deg_dir):
    input_files = glob.glob(f"{deg_dir}/**/*.wav", recursive=True)
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")

    stoi_scores = []
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.relpath(deg_wav, deg_dir) )
        rate, ref = wavfile.read(ref_wav)
        rate, deg = wavfile.read(deg_wav)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        cur_stoi = stoi(ref, deg, rate, extended=False)
        stoi_scores.append(cur_stoi)

    return np.mean(stoi_scores)



def test_batch():
    args = get_parser().parse_args()
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")


    input_lists = sorted(glob.glob(str(args.input) +'/**/*.wav', recursive=True))

    soundstream = SoundStream(n_filters=32, D=512, ratios=args.ratios)
    parameter_dict = torch.load(args.resume_path)
    new_state_dict = OrderedDict()
    for k, v in parameter_dict.items(): # k is module.xxx.weight, v is weight
        name = k[7:] # truncate the xxx.weight after `module.`
        new_state_dict[name] = v
    soundstream.load_state_dict(new_state_dict) # load model
    soundstream = soundstream.cuda()
    os.makedirs(args.output, exist_ok=True)
    for audio in input_lists:
        test_one(os.path.join(args.input,audio), os.path.join(args.output,os.path.relpath(audio, str(args.input)) ), args.rescale, args, soundstream)
 
    nb_score, wb_score = cal_pesq(args.input, args.output, args.samplerate)
    print(f"NB PESQ: {nb_score}")
    print(f"WB PESQ: {wb_score}")

    stoi_score = calculate_stoi(args.input, args.output)
    print(f"STOI: {stoi_score}")

if __name__ == '__main__':
    #main()
    test_batch()
