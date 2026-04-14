"""Touchstone read/write (Hz, S RI, 50 Ω)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_touchstone_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a 2-port Touchstone ``.s2p`` file (RI format).

    Returns
    -------
    freqs_hz : ndarray, shape (F,)
    sparams  : ndarray, shape (F, 2, 2), complex128
    """
    path = Path(path)
    freqs: list[float] = []
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            vals = [float(v) for v in parts]
            if len(vals) == 9:
                # Full row: freq s11r s11i s21r s21i s12r s12i s22r s22i
                freqs.append(vals[0])
                rows.append(vals[1:])
            elif len(vals) == 4 and rows:
                # Continuation line (s12r s12i s22r s22i)
                rows[-1].extend(vals)
    freqs_hz = np.array(freqs)
    n = len(freqs_hz)
    sparams = np.zeros((n, 2, 2), dtype=np.complex128)
    for idx, row in enumerate(rows):
        sparams[idx, 0, 0] = row[0] + 1j * row[1]
        sparams[idx, 1, 0] = row[2] + 1j * row[3]
        sparams[idx, 0, 1] = row[4] + 1j * row[5]
        sparams[idx, 1, 1] = row[6] + 1j * row[7]
    return freqs_hz, sparams


def read_touchstone_snp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read an N-port Touchstone ``.sNp`` file (RI format).

    Returns
    -------
    freqs_hz : ndarray, shape (F,)
    sparams  : ndarray, shape (F, N, N), complex128
    """
    path = Path(path)
    # Determine number of ports from extension
    ext = path.suffix.lower()
    if ext.startswith(".s") and ext.endswith("p"):
        n_ports = int(ext[2:-1])
    else:
        raise ValueError(f"Cannot determine port count from extension: {ext}")

    n_values_per_freq = n_ports * n_ports * 2  # real + imag per S-param

    freqs: list[float] = []
    data_rows: list[list[float]] = []
    current_values: list[float] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            vals = [float(v) for v in parts]

            if not current_values:
                # New frequency point: first value is frequency
                freqs.append(vals[0])
                current_values.extend(vals[1:])
            else:
                # Continuation line
                current_values.extend(vals)

            if len(current_values) >= n_values_per_freq:
                data_rows.append(current_values[:n_values_per_freq])
                current_values = []

    freqs_hz = np.array(freqs)
    n = len(freqs_hz)
    sparams = np.zeros((n, n_ports, n_ports), dtype=np.complex128)
    for idx, row in enumerate(data_rows):
        k = 0
        for i in range(n_ports):
            for j in range(n_ports):
                sparams[idx, i, j] = row[k] + 1j * row[k + 1]
                k += 2
    return freqs_hz, sparams


def write_touchstone_nport(
    output_path: Path,
    num_ports: int,
    freqs_hz: np.ndarray,
    sparams: np.ndarray,
) -> None:
    """
    Write n-port Touchstone (.sNp). ``sparams`` shape (F, N, N), complex.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("# Hz S RI R 50\n")
        f.write("! Written by invdesign.touchstone\n")
        f.write(f"! NPORTS = {num_ports}, NPOINTS = {len(freqs_hz)}\n")

        for idx, freq in enumerate(freqs_hz):
            f.write(f"{freq:.6e}")
            values_written_in_line = 0
            for i in range(num_ports):
                for j in range(num_ports):
                    sij = sparams[idx, i, j]
                    if values_written_in_line >= 4:
                        f.write("\n ")
                        values_written_in_line = 0
                    f.write(f" {sij.real:.6e} {sij.imag:.6e}")
                    values_written_in_line += 1
            f.write("\n")
