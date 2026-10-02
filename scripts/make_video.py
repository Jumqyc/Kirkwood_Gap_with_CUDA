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
import time

import numpy as np
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from kirkwood_io import (elements, epoch_header, read_epoch,  # noqa: E402
                         resonance_axes, semimajor_axis)
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


# The step ladder depends on the integrator, and not by a little: a
# Wisdom-Holman drift is an exact Kepler advance, so its step is limited by how
# fast the perturbation changes rather than by perihelion, and it takes steps
# about ten times larger for the same accuracy. Running the Yoshida-4 ladder
# with Wisdom-Holman would take the same number of steps at 4.5x the cost per
# step, which is 4.5x slower and not 2.4x faster -- the speedup is entirely in
# the step size, and it has to be spent here.
YOSHIDA4_LADDER = [
    Segment("s1", 0.0, 1_000.0, 0.5, "kirkwood_gpu_dt0p5"),
    Segment("s2", 1_000.0, 10_000.0, 2, "kirkwood_gpu_dt2"),
    Segment("s3", 10_000.0, 100_000.0, 10, "kirkwood_gpu"),
    Segment("s4", 100_000.0, 1_000_000.0, 20, "kirkwood_gpu_dt20"),
]

# 200 days is T_Jupiter / 20, the step the reference implementation uses.
WISDOM_HOLMAN_LADDER = [
    Segment("s1", 0.0, 1_000.0, 5, "kirkwood_gpu_dt5"),
    Segment("s2", 1_000.0, 10_000.0, 20, "kirkwood_gpu_dt20"),
    Segment("s3", 10_000.0, 100_000.0, 100, "kirkwood_gpu_dt100"),
    Segment("s4", 100_000.0, 1_000_000.0, 200, "kirkwood_gpu_dt200"),
]



class Stalled(RuntimeError):
    """A batch produced more steps without writing a dump for too long."""


def newest_dump_mtime(dumps: Path) -> float:
    """Modification time of the most recently written epoch file, or 0."""
    files = list(dumps.glob("epoch_*.bin"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def run_binary(binary: Path, n_particle: int, n_step: int, dumps: Path,
               epoch_every: int, seed: int, stall_seconds: float,
               wisdom_holman: bool) -> None:
    """Runs one batch, killing it if it stops making progress.

    Args:
        binary: the integrator to launch.
        n_particle, n_step, epoch_every, seed: passed through to the binary.
        dumps: directory the binary writes epochs into; watched for progress.
        wisdom_holman: tenth argument to the binary; true selects the
            Wisdom-Holman mapping over the Yoshida-4 composition. Measured 2.4x
            faster overall at 1e6 particles, from 10.8x larger steps against a
            4.5x higher cost per step.
        stall_seconds: give up if no new epoch appears for this long. Every
            segment writes one at least every ~95 s -- the dt = 20 binary at
            4e6 particles is the slowest -- so a gap this long is a hang, not
            a slow step.

    A hang is not something the binary can report. cuda/gpu_simulation.cu
    checks every CUDA call, but a call that never returns has no error code to
    check, and the process sits in cudaDeviceSynchronize burning a core with
    the GPU idle. That is exactly what happened once and cost eight hours.

    Raises:
        Stalled: if the batch made no progress for `stall_seconds`.
    """
    cmd = [str(binary), str(n_particle), str(n_step), str(dumps),
           str(epoch_every), "0.1", "1", str(seed), "1",
           "1" if wisdom_holman else "0"]
    print("    $ " + " ".join(cmd[1:]), flush=True)

    # The poll interval is the floor on what this can detect: a stall shorter
    # than POLL cannot be seen, because any poll that finds a new epoch resets
    # the deadline. The default threshold is 300 s against a 5 s poll, so the
    # margin is wide; a threshold below POLL would never fire at all.
    POLL = 5.0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    last = newest_dump_mtime(dumps)
    deadline = time.monotonic() + stall_seconds
    while proc.poll() is None:
        time.sleep(POLL)
        now = newest_dump_mtime(dumps)
        if now > last:
            last = now
            deadline = time.monotonic() + stall_seconds
        elif time.monotonic() > deadline:
            proc.kill()
            proc.wait(timeout=30)
            raise Stalled(f"no new epoch for {stall_seconds:.0f} s at step {n_step}")

    out = proc.stdout.read() if proc.stdout else ""
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out)
    # The binary prints which epoch it resumed from. That line is the only
    # evidence the segments are actually chained, so it has to reach the log.
    for line in out.splitlines():
        if "resuming" in line:
            print("      " + line.strip(), flush=True)


def build_segment(seg: Segment, build: Path, work: Path, frames: int,
                  chunk: int, n_particle: int, seed: int, vmax: float,
                  stall_seconds: float, attempts: int,
                  keep_dumps: bool, wisdom_holman: bool) -> list[Path]:
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
        # A stall costs only the current batch: the state is on disk, so the
        # retry resumes from the newest epoch and the steps taken are not redone.
        for attempt in range(1, attempts + 1):
            try:
                run_binary(build / seg.binary, n_particle, n_step, dumps,
                           epoch_every, seed, stall_seconds, wisdom_holman)
                break
            except (Stalled, subprocess.CalledProcessError) as exc:
                print(f"    !! attempt {attempt}/{attempts} failed: {exc}", flush=True)
                if attempt == attempts:
                    raise

        if baseline is None:
            baseline = load_baseline(work / "baseline.npy", dumps)

        # Only this segment's epochs. The dump directory is shared, so a restart
        # finds the later segments' dumps sitting in it, and rendering those into
        # this segment's frame directory is silent -- the frames look fine and
        # are simply at the wrong times. Compared by step * dt_days, not by step:
        # a step is dt days, so step numbers from different segments are not
        # comparable and the ranges overlap.
        def mine(path: Path) -> bool:
            step, dt_days = epoch_header(path)
            years = step * dt_days / DAYS_PER_YEAR
            return seg.start_year - 1.0 <= years <= seg.end_year + 1.0

        pending = [p for p in epoch_paths(dumps)
                   if mine(p) and not png_of(p).exists()]
        if not pending:
            print("    (nothing new to render)")
            continue

        ep = read_epoch(pending[0])
        resonances = resonance_axes(
            semimajor_axis(ep.planet_r[0], ep.planet_v[0], ep.G))
        for path in pending:
            render(path, frames_dir, resonances, baseline, vmax)
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
    parser.add_argument("--vmax", type=float, default=200.0,
                        help="top of the density colour scale; scale it with "
                             "n_particle or the dense core saturates")
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--only", default=None, help="build one segment, by name")
    parser.add_argument("--wh", action="store_true",
                        help="integrate with Wisdom-Holman rather than Yoshida-4")
    parser.add_argument("--stall-seconds", type=float, default=300.0,
                        help="kill and retry a batch that writes no epoch for "
                             "this long; every segment writes one at least "
                             "every ~95 s, so the default is a hang detector")
    parser.add_argument("--attempts", type=int, default=3,
                        help="tries per batch before giving up")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()

    # One decision, not two: the step ladder follows the integrator, so they
    # cannot be chosen inconsistently.
    segments = WISDOM_HOLMAN_LADDER if args.wh else YOSHIDA4_LADDER
    chosen = [s for s in segments if args.only is None or s.name == args.only]
    if not chosen:
        print(f"no segment named {args.only}", file=sys.stderr)
        return 1

    args.work.mkdir(parents=True, exist_ok=True)
    all_frames: list[Path] = []
    for n, seg in enumerate(segments):
        if seg not in chosen:
            continue
        frames = build_segment(seg, args.build, args.work, args.frames,
                               args.chunk, args.n_particle, args.seed,
                               vmax=args.vmax, stall_seconds=args.stall_seconds,
                               attempts=args.attempts, keep_dumps=args.keep_dumps,
                               wisdom_holman=args.wh)
        all_frames.extend(frames)

    if args.only is None and not args.no_encode:
        encode(all_frames, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
