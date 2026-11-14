import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

SR = 16000
N = 32000
F1, F2 = 1600.0, 2400.0   # trigger tones found in the failures
DUR = 0.07                # length of each tone (s)
GAP = 0.34                # time between tone starts (s)
ENERGY = 0.0095           # 0.95% of the clip energy, limit is 1%
MARGIN = int(0.02 * SR)

n = int(DUR * SR)
t = np.arange(n) / SR
w = tukey(n, 0.15)        # soft fade in/out, no clicks
TONE1 = w * np.sin(2 * np.pi * F1 * t)
TONE2 = w * np.sin(2 * np.pi * F2 * t)
G = int(GAP * SR)
SOS = butter(4, [1300, 2700], btype="band", fs=SR, output="sos")


def make_trigger(x):
    # put the two tones where the clip is quietest around 1.3-2.7 kHz
    band = sosfiltfilt(SOS, x) ** 2
    c = np.concatenate(([0.0], np.cumsum(band)))
    starts = np.arange(MARGIN, N - G - n - MARGIN, 160)
    cost = (c[starts + n] - c[starts]) + (c[starts + G + n] - c[starts + G])
    a = int(starts[np.argmin(cost)])
    y = np.zeros(N)
    y[a:a + n] += TONE1
    y[a + G:a + G + n] += TONE2
    return y


def transform(audio):
    audio = np.asarray(audio, dtype=np.float32)
    out = np.empty_like(audio)
    for i, row in enumerate(audio):
        x = row.astype(np.float64)
        energy = np.sum(x ** 2)
        if energy <= 0:
            out[i] = row
            continue
        y = make_trigger(x)
        scale = np.sqrt(ENERGY * energy / np.sum(y ** 2))
        z = x + scale * y
        while np.max(np.abs(z)) >= 0.999 and scale > 1e-9:   # avoid clipping
            scale *= 0.8
            z = x + scale * y
        if np.max(np.abs(z)) >= 0.999:
            z = x
        out[i] = z.astype(np.float32)
    return out


def main():
    with np.load("test-input.npz", allow_pickle=False) as data:
        audio = data["audio"]
        sample_rate = data["sample_rate"]
        clip_ids = data["clip_ids"]
    np.savez("test-output.npz", audio=transform(audio),
             sample_rate=sample_rate, clip_ids=clip_ids)


if __name__ == "__main__":
    main()
