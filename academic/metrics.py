import librosa
from pystoi import stoi
from pesq import pesq
import numpy as np
import sys


def check_clipping(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr,
        )


class QualityScores:
    def __init__(
        self,
        metric_name,
        rescale,
        force_resample,
        sample_rate=1600,
        target_sr=1600,
    ):
        self.sample_rate = sample_rate
        self.rescale = rescale
        self.metric_name = metric_name
        self.force_resample = force_resample
        self.target_sr = target_sr

        assert metric_name in ["stoi", "pesq"], print(
            f"unknown metric mame {metric_name}"
        )

    def preprocess(self, original, decoded):
        check_clipping(decoded, self.rescale)

        sr = self.sample_rate

        if self.force_resample:
            if self.sample_rate != self.target_sr:
                original = librosa.resample(
                    original, orig_sr=self.sample_rate, target_sr=self.target_sr
                )
                decoded = librosa.resample(
                    decoded, orig_sr=self.sample_rate, target_sr=self.target_sr
                )
                sr = self.target_sr

        min_len = min(len(original), len(decoded))
        original = original[:min_len]
        decoded = decoded[:min_len]

        return original, decoded, sr

    def _stoi(self, original, decoded, samplerate):
        return stoi(original, decoded, samplerate, extended=False)

    def _pesq(self, original, decoded, samplerate):
        return pesq(samplerate, original, decoded, "nb")

    def __call__(self, original, decoded):
        original, decoded, sr = self.preprocess(original, decoded)

        if self.metric_name == "stoi":
            return self._stoi(original, decoded, sr)

        if self.metric_name == "pesq":
            return self._pesq(original, decoded, sr)

    def batch_process(self, original, decoded, agg="sum"):
        metric_values = []

        assert agg in ["sum", "mean"], f"unkown aggregation was provided: {agg}"

        for orig, dec in zip(original, decoded):
            orig, dec = (
                orig.squeeze(0),
                dec.squeeze(0),
            )
            metric_values.append(self.__call__(orig, dec))

        if agg == "sum":
            return np.sum(metric_values)

        elif agg == "mean":
            return np.mean(metric_values)
