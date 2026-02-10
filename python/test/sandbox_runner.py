#!/usr/bin/env python3
import subprocess
import numpy as np
import os

class FlowtreeSandboxRunner:
    """
    C++ 砂場バイナリをサブプロセスで呼び出し、Top-k (ids, scores) を取得するラッパ。
    出力は "doc_id\\tscore" を k 行。
    """
    def __init__(self, exe_path: str, data_dir: str):
        self.exe = os.path.abspath(exe_path)
        self.data_dir = os.path.abspath(data_dir)
        if not os.path.exists(self.exe):
            raise FileNotFoundError(f"flowtree_sandbox not found: {self.exe}")
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        # 必須ファイルの存在チェック
        for fname in ("vocab.npy", "dataset.npz", "query.npz", "input_ids.npy"):
            f = os.path.join(self.data_dir, fname)
            if not os.path.exists(f):
                raise FileNotFoundError(f"missing: {f}")

    def topk(self, k: int):
        cmd = [self.exe, self.data_dir, str(int(k))]
        out = subprocess.check_output(cmd, text=True)
        ids, scores = [], []
        for line in out.strip().splitlines():
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"unexpected output line: {line!r}")
            di, sc = parts
            ids.append(int(di))
            scores.append(float(sc))
        return np.array(ids, dtype=np.int32), np.array(scores, dtype=np.float32)
