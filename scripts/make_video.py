"""Builds the Kirkwood video: three segments, three step sizes, one MP4.

The orchestration lives here rather than in the binaries or in a shell script.
The binary knows how to run N steps and write epochs; it does not know that the
video is cut into segments, that they use different dt, or that the intermediate
dumps should be deleted as they are consumed.

The three segments

    0 - 10 kyr      dt = 2 days     the spikes grow; nothing is static yet
    10 - 100 kyr    dt = 10 days    the 5:2 finishes growing
    100 kyr - 1 Myr dt = 20 days    nothing is changing, so a coarse step will do

are separate resolutions of the same model, and they chain: each starts from the
last epoch the previous one wrote. That works only because the epoch format
records its own dt and the reader converts out of it -- see read_epoch.

Each segment runs in rounds. A round advances the simulation by CHUNK frames'
worth of steps, renders those frames, and deletes every dump but the newest, so
peak disk use is one round rather than one segment.

Usage:
    python scripts/make_video.py [--out data/video/kirkwood.mp4]
                                 [--frames 600] [--chunk 100]
                                 [--only s1] [--keep-dumps]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from kirkwood_io import elements, read_epoch, resonance_axes, semimajor_axis  # noqa: E402
from render_frames import render, epoch_paths, load_baseline  # noqa: E402

DAYS_PER_YEAR = 365.25
FPS = 30


@dataclass
class Segment:
    """One resolution of the model, and the slice of time it covers."""

    name: str
    start_year: float
    end_year: float
    dt_days: float
    binary: str

    @property
    def steps_at_start(self) -> int:
        return round(self.start_year * DAYS_PER_YEAR / self.dt_days)

    @property
    def steps_at_end(self) -> int:
        return round(self.end_year * DAYS_PER_YEAR / self.dt_days)


SEGMENTS = [
    Segment("s1", 0.0, 1_000.0, 0.5, "kirkwood_gpu_dt0p5"),
    Segment("s2", 1_000.0, 10_000.0, 2, "kirkwood_gpu_dt2"),
    Segment("s3", 10_000.0, 100_000.0, 10, "kirkwood_gpu"),
    Segment("s4", 100_000.0, 1_000_000.0, 20, "kirkwood_gpu_dt20"),
]


def run_binary(binary: Path, n_particle: int, n_step: int, dumps: Path,
               epoch_every: int, seed: int) -> None:
    """Runs one batch. `n_step` is the total step count in this binary's dt."""
    cmd = [str(binary), str(n_particle), str(n_step), str(dumps),
           str(epoch_every), "0.1", "1", str(seed), "1"]
    print("    $ " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, capture_output=True)


def build_segment(seg: Segment, build: Path, work: Path, frames: int,
                  chunk: int, n_particle: int, seed: int,
                  keep_dumps: bool) -> list[Path]:
    """Runs one segment in rounds and returns its rendered frames, in order."""
    # One dump directory for every segment, because that is the channel the
    # segments hand the state to each other through: segment k resumes from the
    # newest epoch segment k-1 left behind. Frame directories stay separate, so
    # each segment's frames keep their own ordering.
    dumps = work / "dumps"
    frames_dir = work / seg.name / "frames"
    dumps.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    epoch_every = max(1, (seg.steps_at_end - seg.steps_at_start) // frames)
    print(f"\n{seg.name}: {seg.start_year / 1000:.0f}-{seg.end_year / 1000:.0f} kyr,"
          f" dt = {seg.dt_days} d, {frames} frames, epoch_every = {epoch_every} steps")
    if seg.steps_at_start:
        print(f"  resumes from whatever {work} holds; expects to start near "
              f"step {seg.steps_at_start}")

    # The baseline is the t = 0 distribution, which the right-hand panel divides
    # by. It is cached to .npy because the loop below deletes the epoch it came
    # from, and it cannot be read at all until a round has written that epoch.
    baseline: np.ndarray | None = None

    def png_of(path: Path) -> Path:
        return frames_dir / f"frame_{int(path.stem.split('_')[1]):09d}.png"

    done = 0
    while done < frames:
        done = min(frames, done + chunk)
        n_step = seg.steps_at_start + done * epoch_every
        run_binary(build / seg.binary, n_particle, n_step, dumps, epoch_every, seed)

        if baseline is None:
            baseline = load_baseline(work / "baseline.npy", dumps)

        pending = [p for p in epoch_paths(dumps) if not png_of(p).exists()]
        if not pending:
            print("    (nothing new to render)")
            continue

        ep = read_epoch(pending[0])
        resonances = resonance_axes(
            semimajor_axis(ep.planet_r[0], ep.planet_v[0], ep.G))
        for path in pending:
            render(path, frames_dir, resonances, baseline)
        print(f"    round {done // chunk}: rendered {len(pending)} frames,"
              f" {done}/{frames} in this segment")

        if not keep_dumps and len(epoch_paths(dumps)) > 1:
            for path in epoch_paths(dumps)[:-1]:
                path.unlink()

    return sorted(frames_dir.glob("frame_*.png"))


def encode(frames: list[Path], out: Path) -> None:
    """Joins the frames into an MP4. Frames are already in playing order."""
    listing = out.parent / (out.stem + "_frames.txt")
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in frames))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-r", str(FPS), "-i", str(listing),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium", str(out),
    ], check=True)
    listing.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/video/kirkwood.mp4")
    parser.add_argument("--work", type=Path, default=ROOT / "data/video/work")
    parser.add_argument("--build", type=Path, default=ROOT / "build-o3")
    parser.add_argument("--frames", type=int, default=600, help="per segment")
    parser.add_argument("--chunk", type=int, default=100,
                        help="frames per round, i.e. the disk bound")
    parser.add_argument("--n-particle", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--only", default=None, help="build one segment, by name")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()

    chosen = [s for s in SEGMENTS if args.only is None or s.name == args.only]
    if not chosen:
        print(f"no segment named {args.only}", file=sys.stderr)
        return 1

    args.work.mkdir(parents=True, exist_ok=True)
    all_frames: list[Path] = []
    for n, seg in enumerate(SEGMENTS):
        if seg not in chosen:
            continue
        frames = build_segment(seg, args.build, args.work, args.frames,
                               args.chunk, args.n_particle, args.seed,
                               keep_dumps=args.keep_dumps)
        all_frames.extend(frames)

    if args.only is None and not args.no_encode:
        encode(all_frames, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
