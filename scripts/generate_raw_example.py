"""Generate a deterministic Bayer test chart for the P1 reader demo."""

from pathlib import Path

import numpy as np


def main() -> None:
    y, x = np.mgrid[0:192, 0:320]
    rgb = np.stack((x / 319, y / 191, ((x // 40 + y // 48) % 2) * 0.45 + 0.05), axis=-1)
    samples = np.rint(64 + rgb * (4095 - 64)).astype("<u2")
    mosaic = samples[..., 1].copy()
    mosaic[0::2, 0::2] = samples[0::2, 0::2, 0]
    mosaic[1::2, 1::2] = samples[1::2, 1::2, 2]
    output = Path("data/synthetic/p1_raw12.rggb")
    output.parent.mkdir(parents=True, exist_ok=True)
    mosaic.tofile(output)
    print(output)


if __name__ == "__main__":
    main()
