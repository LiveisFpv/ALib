#!/usr/bin/env python3
"""Build a FAISS index from generated embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def load_memmap(mem_path: Path) -> tuple[np.memmap, int, int]:
    meta_path = mem_path.with_suffix(".shape.json")
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        n_vectors, dim = json.load(meta_file)
    arr = np.memmap(mem_path, dtype=np.float16, mode="r", shape=(n_vectors, dim))
    return arr, int(n_vectors), int(dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--emb-dir", required=True, help="Directory containing embedding outputs")
    parser.add_argument("--memfile", default="doc_embeddings.f16.memmap", help="Memmap file name")
    parser.add_argument("--doc-ids", default="doc_ids.npy", help="Doc IDs numpy file")
    parser.add_argument("--out", help="Output index path (defaults to <emb-dir>/faiss.index)")
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="ivfpq")
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    mem_path = emb_dir / args.memfile
    doc_ids_path = emb_dir / args.doc_ids
    out_path = Path(args.out) if args.out else emb_dir / "faiss.index"

    embeddings, n_vecs, dim = load_memmap(mem_path)
    doc_ids = np.load(doc_ids_path)
    if len(doc_ids) != n_vecs:
        raise SystemExit("Document IDs count does not match embeddings count")

    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2

    rng = np.random.RandomState(args.seed)
    train_size = min(args.train_size, n_vecs)
    train_idx = rng.choice(n_vecs, size=train_size, replace=False)
    train_vectors = np.asarray(embeddings[train_idx], dtype=np.float32)

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)
    else:
        description = f"IVF{args.nlist},PQ{args.m}x{args.nbits}"
        index = faiss.index_factory(dim, description, metric)
        index.train(train_vectors)

    for start in tqdm(range(0, n_vecs, 200_000), desc="Adding", unit="vecs", dynamic_ncols=True):
        end = min(start + 200_000, n_vecs)
        vectors = np.asarray(embeddings[start:end], dtype=np.float32)
        index.add(vectors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    meta = {
        "vectors": n_vecs,
        "dimension": dim,
        "metric": args.metric,
        "type": args.index_type,
        "nlist": getattr(index, "nlist", None),
        "doc_ids": str(doc_ids_path.name),
    }
    with open(out_path.with_suffix(out_path.suffix + ".meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    print(f"Saved index to {out_path}")


if __name__ == "__main__":
    main()
