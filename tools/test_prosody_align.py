"""Atea el muestreo de prosodia de hefesto.prepare_prosody_data.

Caso A: prosodia ya alineada a 1 fila por frame de video (lo que entrega el
   pipeline de parole: spline a los timestamps de frame). No hay que tirar nada.
Caso B: prosodia densa, varios muestreos por frame (de donde viene el ::6
   historico). Hay que quedarse con uno por frame, sin importar la posicion.

El codigo actual usa iloc[::6], que solo acierta por casualidad si la densidad
es exactamente 6 y las filas vienen ordenadas.
"""
import os
import sys
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from hefesto import prepare_prosody_data  # noqa: E402


def write_parquet(rows):
    fd, path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def base_cols(n, frame_fn):
    return {
        "Frame": [frame_fn(i) for i in range(n)],
        "pitch": [100.0 + i for i in range(n)],
        "intensity": [60.0 + i for i in range(n)],
        "harmonicity": [0.5] * n,
    }


def case_one_row_per_frame():
    path = write_parquet(base_cols(10, lambda i: i))
    out = prepare_prosody_data(path)
    os.unlink(path)
    frames = sorted(out["frame"].tolist())
    ok = frames == list(range(10))
    print(f"[{'ok ' if ok else 'FAIL'}] 1 fila/frame -> {len(out)} filas, frames {frames}")
    return ok


def case_dense_six_per_frame():
    # 6 muestras por frame, frames 0..9 intercalados
    path = write_parquet(base_cols(60, lambda i: i // 6))
    out = prepare_prosody_data(path)
    os.unlink(path)
    frames = sorted(out["frame"].tolist())
    ok = frames == list(range(10)) and len(out) == 10
    print(f"[{'ok ' if ok else 'FAIL'}] denso 6/frame -> {len(out)} filas, frames {frames}")
    return ok


def case_dense_unsorted():
    # denso y ADEMAS barajado: el ::6 posicional elige una submuestra arbitraria
    import random
    rows = base_cols(60, lambda i: i // 6)
    idx = list(range(60))
    random.Random(0).shuffle(idx)
    rows = {k: [v[j] for j in idx] for k, v in rows.items()}
    path = write_parquet(rows)
    out = prepare_prosody_data(path)
    os.unlink(path)
    covered = set(out["frame"].tolist())
    ok = covered == set(range(10))
    print(f"[{'ok ' if ok else 'FAIL'}] denso desordenado -> cubre {len(covered)}/10 frames")
    return ok


if __name__ == "__main__":
    results = [case_one_row_per_frame(), case_dense_six_per_frame(), case_dense_unsorted()]
    print("\nTOTAL:", sum(results), "/", len(results))
    sys.exit(0 if all(results) else 1)
