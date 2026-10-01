"""Read the epoch files written by ``Simulation::dump``.

The binary layout is defined exactly once, in the doc comment of
``Simulation::dump`` in ``cpp/simulation.hpp``; this module only consumes it.
In particular it never re-derives a physical constant -- ``dt`` and ``G`` are
read from the file header, because a constant written down twice is a constant
that will eventually disagree with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import struct

import numpy as np
import numpy.typing as npt

MAGIC = 0x4B49524B
VERSION = 1

# u32 magic, u32 version, u64 n_planet, f64 dt, f64 G
HEADER_BYTES = 4 + 4 + 8 + 8 + 8
# u64 n_alive, u64 step
RECORD_FIXED_BYTES = 8 + 8

_FLOATS_PER_BODY = 6

# Mean-motion resonances of the form (p + q) : p -- the particle completes p + q
# orbits while the planet completes p. Each entry is (p, q).
_RESONANCES: dict[str, tuple[int, int]] = {
    "4:1": (1, 3),
    "3:1": (1, 2),
    "5:2": (2, 3),
    "7:3": (3, 4),
    "2:1": (1, 1),
}


@dataclass(frozen=True)
class Epoch:
    """One snapshot of the simulation, as written by a single ``dump`` call.

    Attributes:
        step: Integration steps taken so far, in units of ``dt_days`` days.
        dt_days: Step size in days, from the file header.
        G: Gravitational constant in AU^3 / (M_sun * step^2). It is per *step*,
            not per day, matching the velocities below. Any orbital-element
            formula is consistent with this as long as it uses `v` unchanged,
            because the step unit cancels.
        planet_r: ``(n_planet, 3)`` massive-body positions in AU, heliocentric.
        planet_v: ``(n_planet, 3)`` massive-body velocities in AU per step.
        r: ``(n_alive, 3)`` test-particle positions in AU, heliocentric.
        v: ``(n_alive, 3)`` test-particle velocities in AU per step.
    """

    step: int
    dt_days: float
    G: float
    planet_r: npt.NDArray[np.float64]
    planet_v: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]

    @property
    def n_alive(self) -> int:
        """Number of test particles that survived to this epoch."""
        return int(self.r.shape[0])

    @property
    def time_years(self) -> float:
        """Simulated time since the start of the run, in years."""
        return self.step * self.dt_days / 365.0


# magic, version, n_planet, dt_days, G, n_alive, step.
_HEADER = struct.Struct("<IIQddQQ")


def epoch_header(path: str | Path) -> tuple[int, float]:
    """Reads only the fixed-size header of an epoch file.

    Args:
        path: an epoch file written by write_epoch.
    Returns:
        (step, dt_days) -- the step count and the step size it is counted in.
    Raises:
        ValueError: if the magic or version does not match.

    Cheap enough to call on every file in a directory: the header is a few dozen
    bytes of a file that may be hundreds of megabytes. The step count alone is
    not comparable across files, because a step is dt days; step * dt_days is.
    """
    with open(path, "rb") as f:
        magic, version, n_planet, dt_days, _G, n_alive, step = _HEADER.unpack(
            f.read(_HEADER.size))
    if magic != 0x4B49524B or version != 1:
        raise ValueError(f"{path}: not an epoch file (magic={magic:#x} version={version})")
    return step, dt_days


def read_epoch(path: str | Path) -> Epoch:
    """Read one ``epoch_<step>.bin`` file into arrays.

    Args:
        path: File written by ``Simulation::dump``.
    Returns:
        The epoch. Every array owns its memory, so the file is closed on
        return.
    Raises:
        ValueError: If the magic or the version is unrecognised, if the file
            size disagrees with the counts in its own header, or if any read
            comes up short. A run killed mid-write leaves exactly that, and
            ``numpy.fromfile`` returns a short array rather than failing.
    """
    path = Path(path)
    with open(path, "rb") as stream:

        def take(
            dtype: npt.DTypeLike, count: int
        ) -> npt.NDArray[np.generic]:
            """Read exactly `count` scalars; fromfile short-reads at EOF."""
            values = np.fromfile(stream, dtype=dtype, count=count)
            if values.size != count:
                raise ValueError(
                    f"{path}: truncated; wanted {count} x "
                    f"{np.dtype(dtype).name}, got {values.size}"
                )
            return values

        magic, version = take(np.uint32, 2)
        if magic != MAGIC:
            raise ValueError(
                f"{path}: bad magic 0x{magic:08X}, expected 0x{MAGIC:08X}"
            )
        if version != VERSION:
            raise ValueError(
                f"{path}: version {version}, this reader knows {VERSION}"
            )

        (n_planet,) = take(np.uint64, 1)
        dt_days, G = take(np.float64, 2)
        n_alive, step = take(np.uint64, 2)

        expected_bytes = (
            HEADER_BYTES
            + RECORD_FIXED_BYTES
            + _FLOATS_PER_BODY * 8 * (int(n_planet) + int(n_alive))
        )
        on_disk = path.stat().st_size
        if on_disk != expected_bytes:
            raise ValueError(
                f"{path}: {on_disk} bytes on disk but the header implies "
                f"{expected_bytes}; the run was interrupted mid-write, or the "
                f"writer and this reader disagree about the layout"
            )

        # Planets are interleaved in the file: x y z vx vy vz per body.
        planets = take(np.float64, _FLOATS_PER_BODY * int(n_planet)).reshape(
            int(n_planet), _FLOATS_PER_BODY
        )
        # Particles are six consecutive blocks -- the simulator's own SoA
        # layout, so it writes them without transposing.
        blocks = [take(np.float64, int(n_alive)) for _ in range(_FLOATS_PER_BODY)]

    state = np.stack(blocks, axis=1) if blocks else np.empty((0, _FLOATS_PER_BODY))
    return Epoch(
        step=int(step),
        dt_days=float(dt_days),
        G=float(G),
        planet_r=planets[:, :3],
        planet_v=planets[:, 3:],
        r=state[:, :3],
        v=state[:, 3:],
    )


def read_run(dump_dir: str | Path) -> list[Epoch]:
    """Read every epoch file in a directory, oldest first.

    Args:
        dump_dir: Directory holding ``epoch_<step>.bin`` files.
    Returns:
        The epochs sorted by the ``step`` recorded *inside* each file, not by
        filename, so the order never depends on how the names happen to sort.
    Raises:
        FileNotFoundError: If the directory holds no epoch files.
    """
    epochs = [read_epoch(p) for p in sorted(Path(dump_dir).glob("epoch_*.bin"))]
    if not epochs:
        raise FileNotFoundError(f"no epoch_*.bin files in {dump_dir}")
    epochs.sort(key=lambda epoch: epoch.step)
    return epochs


def elements(epoch: Epoch) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Osculating semimajor axis and eccentricity of every test particle.

    .. math::

        a = \left(\frac{2}{|r|} - \frac{|v|^2}{G}\right)^{-1}, \qquad
        \vec e = \frac{\left(|v|^2 - G/|r|\right)\vec r
                      - \left(\vec r \cdot \vec v\right)\vec v}{G}

    These are the two-body elements with respect to the Sun alone: the planet
    perturbs them, which is the entire point of the experiment. A particle on
    a hyperbolic orbit has :math:`a < 0` and is not bound to the Sun.

    Args:
        epoch: Any epoch. Its ``G`` is in per-step units, which cancels
            against the per-step velocities, so no time conversion is needed.
    Returns:
        ``(a, e)``, both of shape ``(n_alive,)``: semimajor axis in AU and
        eccentricity, dimensionless.
    """
    r_norm = np.linalg.norm(epoch.r, axis=1)
    v_sq = np.einsum("ij,ij->i", epoch.v, epoch.v)
    a = 1.0 / (2.0 / r_norm - v_sq / epoch.G)
    e_vec = (
        (v_sq - epoch.G / r_norm)[:, None] * epoch.r
        - np.einsum("ij,ij->i", epoch.r, epoch.v)[:, None] * epoch.v
    ) / epoch.G
    return a, np.linalg.norm(e_vec, axis=1)


def resonance_axes(
    a_planet: float, labels: Sequence[str] | None = None
) -> dict[str, float]:
    """Semimajor axes of the first-order mean-motion resonances.

    A particle at the ``(p + q) : p`` resonance completes ``p + q`` orbits in
    the time the planet completes ``p``, so by Kepler's third law it sits at
    :math:`a = a_{planet} (p / (p + q))^{2/3}`.

    Take `a_planet` from the epoch's own planet state rather than from a
    remembered value: if the perturber is not where you think it is, the
    resonance lines will show it.

    Args:
        a_planet: The perturber's semimajor axis in AU.
        labels: Which resonances to return; defaults to all of them.
    Returns:
        Resonance label -> semimajor axis in AU.
    """
    chosen = _RESONANCES if labels is None else {k: _RESONANCES[k] for k in labels}
    return {
        label: a_planet * (p / (p + q)) ** (2.0 / 3.0)
        for label, (p, q) in chosen.items()
    }


def semimajor_axis(r: npt.NDArray[np.float64], v: npt.NDArray[np.float64], G: float):
    """Semimajor axis of a single body, given position and velocity."""
    r_norm = np.linalg.norm(r)
    return 1.0 / (2.0 / r_norm - float(np.dot(v, v)) / G)
