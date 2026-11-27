#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "h5py"]
# ///
"""Writes the SIFT1M data the benchmark/vss benchmarks read, from ann-benchmarks' sift-128-euclidean.hdf5 (500 MB).

  benchmark/data/sift1m_base.parquet     id, emb: the first REGRESSION_MICRO_VSS_SIFT1M_ROWS (.github/settings.env) base vectors
  benchmark/data/sift1m_queries.parquet  qid, vec: the first 1,000 of the 10,000 test vectors
  benchmark/data/config.benchmark.in     the settings the benchmark/vss templates include
"""
import os
import subprocess

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

URL = 'http://ann-benchmarks.com/sift-128-euclidean.hdf5'
VSS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(VSS_DIR), 'data')
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(VSS_DIR)), '.github', 'settings.env')
with open(SETTINGS_PATH) as settings_file:
    SETTINGS = dict(line.strip().split('=', 1) for line in settings_file if line.strip() and not line.startswith('#'))
ROWS = int(SETTINGS['REGRESSION_MICRO_VSS_SIFT1M_ROWS'])
QUERIES = 1000


def write(path, id_name, vec_name, vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    column = pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), vectors.shape[1])
    pq.write_table(pa.table({id_name: pa.array(np.arange(len(vectors), dtype=np.int64)), vec_name: column}), path)


os.makedirs(DATA_DIR, exist_ok=True)
hdf5_path = os.path.join(DATA_DIR, 'sift-128-euclidean.hdf5')
if not os.path.exists(hdf5_path):
    # curl: ann-benchmarks.com answers Python's urllib with 403.
    subprocess.run(['curl', '-fsSL', '-o', hdf5_path, URL], check=True)
with h5py.File(hdf5_path, 'r') as f:
    if ROWS > len(f['train']):
        raise SystemExit(f"REGRESSION_MICRO_VSS_SIFT1M_ROWS = {ROWS} in settings.env, but SIFT1M has only {len(f['train'])} base vectors")
    write(os.path.join(DATA_DIR, 'sift1m_base.parquet'), 'id', 'emb', f['train'][:ROWS])
    write(os.path.join(DATA_DIR, 'sift1m_queries.parquet'), 'qid', 'vec', f['test'][:QUERIES])
os.remove(hdf5_path)
# benchmark_runner lowercases argument names; override one with benchmark_runner --<name> <value>.
with open(os.path.join(DATA_DIR, 'config.benchmark.in'), 'w') as config:
    config.write(f"argument rows {ROWS}\nargument row_group_size {SETTINGS['REGRESSION_MICRO_VSS_ROW_GROUP_SIZE']}\n")
