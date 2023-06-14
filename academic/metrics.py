import librosa
from pystoi import stoi
from pesq import pesq
import numpy as np


def check_clipping(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


class ValMetric:
    def __init__(metric_name, rescale, force_resample, sample_rate=1600, target_sr=1600)
    
        self.sample_rate = sample_rate
        self.rescale = rescale
        self.metric_name = metric_name
        self.force_resample = force_resample
        self.initial_sr = initial_sr
        self.target_sr = target_sr
        
        assert metric_name in ['stoi','pesq'], print(f'unknown metric mame {metric name}')
        
    def preprocess(self, original, decoded):
        check_clipping(decoded, self.rescale)
        
        sr = self.initisl_sr
        
        if self.force_resample:
            if self.sample_rate != self.target_rate:
                original = librosa.resample(original, orig_sr = self.sample_rate, target_sr= self.target_sr)
                decoded = librosa.resample(decoded, orig_sr = self.sample_rate, target_sr= self.target_sr)
                sr =   self.target_sr

        min_len = min(len(original), len(decoded))
        original = ref[:min_len]
        decoded = deg[:min_len]
        
        return original, decoded, sr
        
    def _stoi(self, original, decoded, samplerate):
        return stoi(original, decoded, samplerate, extended=False)
        
    def _pesq(self, original, decoded, samplerate):
        return pesq(samplerate, original, decoded, 'nb')
    
    def __call__(original, decoded):
       
        original, decoded, sr = self.preprocess(original, decoded)
        
        if self.metric_name == 'stoi':
            return self._stoi(original, decoded, sr)
        
        if self.metric_name == == 'pesq':
             return self._pesq(original, decoded, sr)
         
    def batch_process(original, decoded, agg='sum'):
        metric_values = []
        
        for orig, dec in zip(original, decoded):
            metric_values.append(self.__call__(orig, dec ))
        
        if aff == 'sum':
            return np.sum(metric_values)
        
        elif aff=='mean':
            return np.mean(metric_values)
    
    