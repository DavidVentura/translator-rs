"""Measure Kokoro high-frequency whistle metrics from WAV files.

The metric is phase-insensitive: it compares average STFT power bands rather
than subtracting waveforms. This avoids false positives from small timing/phase
differences between otherwise similar renders.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono PCM16 wav")
        sample_rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    audio = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return sample_rate, audio


def stft_power(audio: np.ndarray, sample_rate: int, nfft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.size < nfft:
        padded = np.zeros(nfft, dtype=np.float64)
        padded[: audio.size] = audio
        audio = padded
    window = np.hanning(nfft)
    frames = []
    for start in range(0, audio.size - nfft + 1, hop):
        frame = audio[start : start + nfft] * window
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    return np.fft.rfftfreq(nfft, 1.0 / sample_rate), np.mean(frames, axis=0)


def band_mean(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(power[mask].mean() + 1e-30)


def band_mean_excluding(
    power: np.ndarray,
    freqs: np.ndarray,
    lo: float,
    hi: float,
    exclude: tuple[tuple[float, float], ...] = (),
) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    for ex_lo, ex_hi in exclude:
        mask &= ~((freqs >= ex_lo) & (freqs < ex_hi))
    return float(power[mask].mean() + 1e-30)


def db(value: float) -> float:
    return 10.0 * float(np.log10(max(value, 1e-30)))


def measure(path: Path, nfft: int, hop: int) -> dict[str, float]:
    sample_rate, audio = read_wav(path)
    freqs, power = stft_power(audio, sample_rate, nfft, hop)

    band48 = band_mean(power, freqs, 4750.0, 4850.0)
    band96 = band_mean(power, freqs, 9550.0, 9650.0)
    neigh48 = (band_mean(power, freqs, 4550.0, 4700.0) + band_mean(power, freqs, 4900.0, 5050.0)) / 2.0
    neigh96 = (band_mean(power, freqs, 9350.0, 9500.0) + band_mean(power, freqs, 9700.0, 9850.0)) / 2.0
    speech = band_mean_excluding(power, freqs, 100.0, 11000.0)
    high = band_mean_excluding(power, freqs, 4000.0, 11000.0)
    high_no_tones = band_mean_excluding(
        power,
        freqs,
        4000.0,
        11000.0,
        exclude=((4720.0, 4880.0), (9520.0, 9680.0)),
    )
    air = band_mean_excluding(power, freqs, 7000.0, 11000.0)
    return {
        "sample_rate": sample_rate,
        "samples": audio.size,
        "seconds": audio.size / sample_rate,
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "speech": speech,
        "high": high,
        "high_no_tones": high_no_tones,
        "air": air,
        "band48": band48,
        "band96": band96,
        "neigh48": neigh48,
        "neigh96": neigh96,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="cmp_fp32_while_prec_low_mnn.wav")
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("wavs", nargs="+")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    ref = measure(ref_path, args.nfft, args.hop)
    rows = [(ref_path, ref)]
    rows.extend((Path(path), measure(Path(path), args.nfft, args.hop)) for path in args.wavs)

    print(f"reference: {ref_path}")
    print(
        "file                                      sec   rms    highD  "
        "highNoToneD airD   4.8tone  9.6tone  4.8D   9.6D"
    )
    for path, item in rows:
        high_delta = db((item["high"] / item["speech"]) / (ref["high"] / ref["speech"]))
        high_no_tone_delta = db(
            (item["high_no_tones"] / item["speech"]) / (ref["high_no_tones"] / ref["speech"])
        )
        air_delta = db((item["air"] / item["speech"]) / (ref["air"] / ref["speech"]))
        tone48 = db(item["band48"] / item["neigh48"])
        tone96 = db(item["band96"] / item["neigh96"])
        delta48 = db((item["band48"] / item["speech"]) / (ref["band48"] / ref["speech"]))
        delta96 = db((item["band96"] / item["speech"]) / (ref["band96"] / ref["speech"]))
        print(
            f"{str(path):42s} {item['seconds']:4.2f} {item['rms']:.5f} "
            f"{high_delta:+6.2f} {high_no_tone_delta:+10.2f} {air_delta:+6.2f} "
            f"{tone48:+8.2f} {tone96:+8.2f} {delta48:+6.2f} {delta96:+6.2f}"
        )


if __name__ == "__main__":
    main()
