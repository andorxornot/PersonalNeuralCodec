import math
import wave
import struct
import os

# Placeholder data generation
def get_sinewave(freq=440.0, duration_ms=200, volume=1.0, sample_rate=44100.0):
  # code adapted from https://stackoverflow.com/a/33913403
  audio = []
  num_samples = duration_ms * (sample_rate / 1000.0)
  for x in range(int(num_samples)):
    audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
  return audio

def save_wav(file_name, audio, sample_rate=44100.0):
  # Open up a wav file
  wav_file=wave.open(file_name,"w")
  # wav params
  nchannels = 1
  sampwidth = 2
  # 44100 is the industry standard sample rate - CD quality.  If you need to
  # save on file size you can adjust it downwards. The stanard for low quality
  # is 8000 or 8kHz.
  nframes = len(audio)
  comptype = "NONE"
  compname = "not compressed"
  wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
  # WAV files here are using short, 16 bit, signed integers for the 
  # sample size.  So we multiply the floating point data we have by 32767, the
  # maximum value for a short integer.  NOTE: It is theortically possible to
  # use the floating point -1.0 to 1.0 data directly in a WAV file but not
  # obvious how to do that using the wave module in python.
  for sample in audio:
      wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
  wav_file.close()
  return

def make_placeholder_dataset(dataset_folder):
  # Make a placeholder dataset with a few .wav files that you can "train" on, just to verify things work e2e
  if os.path.isdir(dataset_folder):
    return
  os.makedirs(dataset_folder)
  save_wav(f"{dataset_folder}/example.wav", get_sinewave())
  save_wav(f"{dataset_folder}/example2.wav", get_sinewave(duration_ms=500))
  os.makedirs(f"{dataset_folder}/subdirectory")
  save_wav(f"{dataset_folder}/subdirectory/example.wav", get_sinewave(freq=330.0))

