#!/usr/bin/env python3
"""Build a FAISS index from generated embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def load_memmap(mem_path: Path) -> tuple[np.memmap, int, int]:
    meta_path = mem_path.with_suffix(".shape.json")
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        n_vectors, dim = json.load(meta_file)
    arr = np.memmap(mem_path, dtype=np.float16, mode="r", shape=(n_vectors, dim))
    return arr, int(n_vectors), int(dim)


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "out1_sum": cache_dir / "cites_out1.sum.f32.memmap",
        "in1_sum": cache_dir / "citedby_in1.sum.f32.memmap",
        "out2_sum": cache_dir / "cites_out2.sum.f32.memmap",
        "in2_sum": cache_dir / "citedby_in2.sum.f32.memmap",
        "out1_mean": cache_dir / "cites_out1.mean.f16.memmap",
        "in1_mean": cache_dir / "citedby_in1.mean.f16.memmap",
        "out2_mean": cache_dir / "cites_out2.mean.f16.memmap",
        "in2_mean": cache_dir / "citedby_in2.mean.f16.memmap",
        "deg_out": cache_dir / "deg_out.npy",
        "deg_in": cache_dir / "deg_in.npy",
    }


def cache_ready(cache_dir: Path, n: int, d: int) -> bool:
    required = ["out1_mean", "in1_mean", "out2_mean", "in2_mean"]
    size = n * d * 2  # float16
    paths = cache_paths(cache_dir)
    for key in required:
        path = paths[key]
        if not path.exists():
            return False
        if path.stat().st_size != size:
            return False
    return True


def iter_edge_batches(paths: list[str], batch_rows: int, desc: str) -> tuple[np.ndarray, np.ndarray]:
    for path in paths:
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows
        pbar = tqdm(total=total, desc=f"{desc}: {os.path.basename(path)}", unit="edge", dynamic_ncols=True)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["source_id", "target_id"])
            src = table.column("source_id").to_numpy(zero_copy_only=False)
            tgt = table.column("target_id").to_numpy(zero_copy_only=False)
            n = len(src)
            for s in range(0, n, batch_rows):
                e = min(s + batch_rows, n)
                pbar.update(e - s)
                yield src[s:e], tgt[s:e]
        pbar.close()


def map_ids(arr: np.ndarray, id_to_idx: dict) -> np.ndarray:
    return np.fromiter((id_to_idx.get(x, -1) for x in arr), dtype=np.int32, count=len(arr))


def zero_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    arr = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    arr[:] = 0
    arr.flush()
    return arr


def write_means(sum_memmap: np.memmap, counts: np.ndarray, out_path: Path, chunk_docs: int) -> None:
    n = sum_memmap.shape[0]
    mean = np.memmap(out_path, dtype=np.float16, mode="w+", shape=sum_memmap.shape)
    for s in tqdm(range(0, n, chunk_docs), desc=f"Write {out_path.name}", unit="doc", dynamic_ncols=True):
        e = min(s + chunk_docs, n)
        chunk = np.array(sum_memmap[s:e], dtype=np.float32, copy=True)
        cnt = counts[s:e].astype(np.float32)
        mask = cnt > 0
        if mask.any():
            chunk[mask] /= cnt[mask][:, None]
        chunk[~mask] = 0.0
        mean[s:e] = chunk.astype(np.float16)
    mean.flush()


def build_citation_cache(
    emb: np.memmap,
    id_to_idx: dict,
    edge_paths: list[str],
    cache_dir: Path,
    *,
    batch_rows: int,
    vec_chunk: int,
    doc_chunk: int,
    keep_sums: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n, d = emb.shape
    paths = cache_paths(cache_dir)

    print("Pass 1/2: level-1 sums")
    out1_sum = zero_memmap(paths["out1_sum"], (n, d))
    in1_sum = zero_memmap(paths["in1_sum"], (n, d))
    deg_out = np.zeros(n, dtype=np.int32)
    deg_in = np.zeros(n, dtype=np.int32)

    for src_ids, tgt_ids in iter_edge_batches(edge_paths, batch_rows=batch_rows, desc="Level1"):
        src_idx = map_ids(src_ids, id_to_idx)
        tgt_idx = map_ids(tgt_ids, id_to_idx)
        mask = (src_idx >= 0) & (tgt_idx >= 0)
        if not mask.any():
            continue
        src_idx = src_idx[mask]
        tgt_idx = tgt_idx[mask]
        if len(src_idx) == 0:
            continue
        np.add.at(deg_out, src_idx, 1)
        np.add.at(deg_in, tgt_idx, 1)

        for s in range(0, len(src_idx), vec_chunk):
            e = min(s + vec_chunk, len(src_idx))
            ss = src_idx[s:e]
            tt = tgt_idx[s:e]
            np.add.at(out1_sum, ss, emb[tt])
            np.add.at(in1_sum, tt, emb[ss])

    out1_sum.flush()
    in1_sum.flush()

    print("Write level-1 means")
    write_means(out1_sum, deg_out, paths["out1_mean"], chunk_docs=doc_chunk)
    write_means(in1_sum, deg_in, paths["in1_mean"], chunk_docs=doc_chunk)

    print("Pass 2/2: level-2 sums")
    out2_sum = zero_memmap(paths["out2_sum"], (n, d))
    in2_sum = zero_memmap(paths["in2_sum"], (n, d))
    cnt_out2 = np.zeros(n, dtype=np.int32)
    cnt_in2 = np.zeros(n, dtype=np.int32)

    for src_ids, tgt_ids in iter_edge_batches(edge_paths, batch_rows=batch_rows, desc="Level2"):
        src_idx = map_ids(src_ids, id_to_idx)
        tgt_idx = map_ids(tgt_ids, id_to_idx)
        mask = (src_idx >= 0) & (tgt_idx >= 0)
        if not mask.any():
            continue
        src_idx = src_idx[mask]
        tgt_idx = tgt_idx[mask]
        if len(src_idx) == 0:
            continue
        np.add.at(cnt_out2, src_idx, deg_out[tgt_idx])
        np.add.at(cnt_in2, tgt_idx, deg_in[src_idx])

        for s in range(0, len(src_idx), vec_chunk):
            e = min(s + vec_chunk, len(src_idx))
            ss = src_idx[s:e]
            tt = tgt_idx[s:e]
            np.add.at(out2_sum, ss, out1_sum[tt])
            np.add.at(in2_sum, tt, in1_sum[ss])

    out2_sum.flush()
    in2_sum.flush()

    print("Write level-2 means")
    write_means(out2_sum, cnt_out2, paths["out2_mean"], chunk_docs=doc_chunk)
    write_means(in2_sum, cnt_in2, paths["in2_mean"], chunk_docs=doc_chunk)

    np.save(paths["deg_out"], deg_out)
    np.save(paths["deg_in"], deg_in)

    if not keep_sums:
        del out1_sum, in1_sum, out2_sum, in2_sum
        for key in ("out1_sum", "in1_sum", "out2_sum", "in2_sum"):
            try:
                paths[key].unlink()
            except OSError:
                pass


def parse_weights(s: str) -> dict[str, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = {"self": 1.0, "out1": 0.0, "in1": 0.0, "out2": 0.0, "in2": 0.0}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad weights token: {p}")
        key, val = p.split("=", 1)
        key = key.strip()
        if key not in out:
            raise ValueError(f"Unknown weight key: {key}")
        out[key] = float(val)
    return out


def build_weighted_vectors(
    idx: np.ndarray,
    emb: np.memmap,
    out1: np.memmap | None,
    in1: np.memmap | None,
    out2: np.memmap | None,
    in2: np.memmap | None,
    weights: dict[str, float],
    *,
    normalize: bool,
) -> np.ndarray:
    vec = weights["self"] * emb[idx].astype(np.float32)
    if out1 is not None and weights["out1"] != 0.0:
        vec += weights["out1"] * out1[idx].astype(np.float32)
    if in1 is not None and weights["in1"] != 0.0:
        vec += weights["in1"] * in1[idx].astype(np.float32)
    if out2 is not None and weights["out2"] != 0.0:
        vec += weights["out2"] * out2[idx].astype(np.float32)
    if in2 is not None and weights["in2"] != 0.0:
        vec += weights["in2"] * in2[idx].astype(np.float32)
    if normalize:
        faiss.normalize_L2(vec)
    return vec


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
    parser.add_argument("--weights", help="Citation weights: self=1.0,out1=0.0,in1=0.0,out2=0.0,in2=0.0")
    parser.add_argument("--edges", nargs="+", help="Parquet edge files with source_id/target_id")
    parser.add_argument("--cache-dir", help="Cache dir for citation vectors (defaults to <emb-dir>/citation_cache)")
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--edge-batch-rows", type=int, default=200_000)
    parser.add_argument("--vec-chunk", type=int, default=10_000)
    parser.add_argument("--doc-chunk", type=int, default=4_096)
    parser.add_argument("--keep-sums", action="store_true")
    parser.add_argument("--add-batch", type=int, default=200_000)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    mem_path = emb_dir / args.memfile
    doc_ids_path = emb_dir / args.doc_ids
    out_path = Path(args.out) if args.out else emb_dir / "faiss.index"

    embeddings, n_vecs, dim = load_memmap(mem_path)
    doc_ids = np.load(doc_ids_path, allow_pickle=True)
    if len(doc_ids) != n_vecs:
        raise SystemExit("Document IDs count does not match embeddings count")

    weights = parse_weights(args.weights) if args.weights else {
        "self": 1.0, "out1": 0.0, "in1": 0.0, "out2": 0.0, "in2": 0.0
    }
    needs_citation = any(weights[k] != 0.0 for k in ("out1", "in1", "out2", "in2"))
    normalize = not args.no_normalize

    out1 = in1 = out2 = in2 = None
    cache_dir = None
    edge_paths: list[str] = []
    if needs_citation:
        if not args.edges:
            raise SystemExit("--edges is required when citation weights are non-zero")
        edge_paths = [str(p) for p in args.edges]
        cache_dir = Path(args.cache_dir) if args.cache_dir else emb_dir / "citation_cache"
        if args.recompute_cache or not cache_ready(cache_dir, n_vecs, dim):
            id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            build_citation_cache(
                embeddings,
                id_to_idx,
                edge_paths,
                cache_dir,
                batch_rows=args.edge_batch_rows,
                vec_chunk=args.vec_chunk,
                doc_chunk=args.doc_chunk,
                keep_sums=args.keep_sums,
            )
        paths = cache_paths(cache_dir)
        out1 = np.memmap(paths["out1_mean"], dtype=np.float16, mode="r", shape=(n_vecs, dim))
        in1 = np.memmap(paths["in1_mean"], dtype=np.float16, mode="r", shape=(n_vecs, dim))
        out2 = np.memmap(paths["out2_mean"], dtype=np.float16, mode="r", shape=(n_vecs, dim))
        in2 = np.memmap(paths["in2_mean"], dtype=np.float16, mode="r", shape=(n_vecs, dim))

    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2

    rng = np.random.RandomState(args.seed)
    train_size = min(args.train_size, n_vecs)
    train_idx = rng.choice(n_vecs, size=train_size, replace=False)
    train_vectors = build_weighted_vectors(
        train_idx, embeddings, out1, in1, out2, in2, weights, normalize=normalize
    )

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)
    else:
        description = f"IVF{args.nlist},PQ{args.m}x{args.nbits}"
        index = faiss.index_factory(dim, description, metric)
        index.train(train_vectors)

    for start in tqdm(range(0, n_vecs, args.add_batch), desc="Adding", unit="vecs", dynamic_ncols=True):
        end = min(start + args.add_batch, n_vecs)
        batch_idx = np.arange(start, end)
        vectors = build_weighted_vectors(
            batch_idx, embeddings, out1, in1, out2, in2, weights, normalize=normalize
        )
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
        "weights": weights,
        "citation_cache": str(cache_dir) if cache_dir else None,
        "citation_edges": edge_paths or None,
        "normalized": normalize,
    }
    with open(out_path.with_suffix(out_path.suffix + ".meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    print(f"Saved index to {out_path}")


if __name__ == "__main__":
    main()
