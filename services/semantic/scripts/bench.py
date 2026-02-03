import os
import json
import argparse
import itertools
import random

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "intfloat/multilingual-e5-large"


def load_shape(mem_path):
    with open(mem_path + ".shape.json", "r", encoding="utf-8") as f:
        n, d = json.load(f)
    return int(n), int(d)


def mean_pool(h, m):
    m = m.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)


@torch.inference_mode()
def embed_queries(tok, mdl, texts, bs=64, max_len=96):
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="Embed queries", unit="batch", dynamic_ncols=True):
        batch = ["query: " + t for t in texts[i:i + bs]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(mdl.device) for k, v in enc.items()}
        h = mdl(**enc).last_hidden_state
        v = mean_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1).cpu().numpy().astype("float32")
        out.append(v)
    return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")


def sample_queries(en_pq, ru_pq, n, seed=0, lang_filter="any"):
    rnd = random.Random(seed)
    picks = []

    def pick_from(path, need_lang):
        if not path:
            return []
        df = pl.read_parquet(path, use_pyarrow=True).select("id", "language", "title")
        if need_lang != "any":
            df = df.filter(pl.col("language") == need_lang)
        rows = list(df.iter_rows(named=True))
        rnd.shuffle(rows)
        return rows

    rows = (
        pick_from(en_pq, "en" if lang_filter != "any" else "any") +
        pick_from(ru_pq, "ru" if lang_filter != "any" else "any")
    )
    rnd.shuffle(rows)
    for r in rows:
        t = (r["title"] or "").strip()
        if not t:
            continue
        picks.append({"id": r["id"], "q": t})
        if len(picks) >= n:
            break
    return picks


def metrics_idx(gt_idx, retrieved_idx, ks):
    n = len(gt_idx)
    out = {}
    for k in ks:
        p_sum = 0.0
        r_sum = 0.0
        mrr_sum = 0.0
        for i in range(n):
            topk = retrieved_idx[i][:k]
            pos = np.where(topk == gt_idx[i])[0]
            if pos.size > 0:
                p_sum += 1.0 / float(k)
                r_sum += 1.0
                mrr_sum += 1.0 / float(pos[0] + 1)
        out[f"P@{k}"] = p_sum / float(n) if n else 0.0
        out[f"R@{k}"] = r_sum / float(n) if n else 0.0
        out[f"MRR@{k}"] = mrr_sum / float(n) if n else 0.0
    return out


def metrics_multi(gt_sets, retrieved_idx, ks, exclude_idx=None):
    n = len(gt_sets)
    out = {}
    for k in ks:
        p_sum = 0.0
        r_sum = 0.0
        mrr_sum = 0.0
        for i in range(n):
            pos_set = gt_sets[i]
            if not pos_set:
                continue
            topk = []
            for idx in retrieved_idx[i]:
                if exclude_idx is not None and idx == exclude_idx[i]:
                    continue
                topk.append(idx)
                if len(topk) >= k:
                    break
            hits = 0
            first_rank = None
            for rank, idx in enumerate(topk, start=1):
                if idx in pos_set:
                    hits += 1
                    if first_rank is None:
                        first_rank = rank
            p_sum += hits / float(k)
            r_sum += hits / float(len(pos_set))
            if first_rank is not None:
                mrr_sum += 1.0 / float(first_rank)
        out[f"P@{k}"] = p_sum / float(n) if n else 0.0
        out[f"R@{k}"] = r_sum / float(n) if n else 0.0
        out[f"MRR@{k}"] = mrr_sum / float(n) if n else 0.0
    return out


def iter_edge_batches(paths, batch_rows=200000, desc="Edges"):
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


def map_ids(arr, id_to_idx):
    return np.fromiter((id_to_idx.get(x, -1) for x in arr), dtype=np.int32, count=len(arr))


def zero_memmap(path, shape):
    arr = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    arr[:] = 0
    arr.flush()
    return arr


def write_means(sum_memmap, counts, out_path, chunk_docs=4096):
    n = sum_memmap.shape[0]
    mean = np.memmap(out_path, dtype=np.float16, mode="w+", shape=sum_memmap.shape)
    for s in tqdm(range(0, n, chunk_docs), desc=f"Write {os.path.basename(out_path)}", unit="doc", dynamic_ncols=True):
        e = min(s + chunk_docs, n)
        chunk = np.array(sum_memmap[s:e], dtype=np.float32, copy=True)
        cnt = counts[s:e].astype(np.float32)
        mask = cnt > 0
        if mask.any():
            chunk[mask] /= cnt[mask][:, None]
        chunk[~mask] = 0.0
        mean[s:e] = chunk.astype(np.float16)
    mean.flush()
    return mean


def cache_paths(cache_dir):
    return {
        "out1_sum": os.path.join(cache_dir, "cites_out1.sum.f32.memmap"),
        "in1_sum": os.path.join(cache_dir, "citedby_in1.sum.f32.memmap"),
        "out2_sum": os.path.join(cache_dir, "cites_out2.sum.f32.memmap"),
        "in2_sum": os.path.join(cache_dir, "citedby_in2.sum.f32.memmap"),
        "out1_mean": os.path.join(cache_dir, "cites_out1.mean.f16.memmap"),
        "in1_mean": os.path.join(cache_dir, "citedby_in1.mean.f16.memmap"),
        "out2_mean": os.path.join(cache_dir, "cites_out2.mean.f16.memmap"),
        "in2_mean": os.path.join(cache_dir, "citedby_in2.mean.f16.memmap"),
        "deg_out": os.path.join(cache_dir, "deg_out.npy"),
        "deg_in": os.path.join(cache_dir, "deg_in.npy"),
        "meta": os.path.join(cache_dir, "cache_meta.json"),
    }


def cache_ready(cache_dir, n, d, meta=None):
    req = [
        "cites_out1.mean.f16.memmap", 
        "citedby_in1.mean.f16.memmap",
        "cites_out2.mean.f16.memmap",
        "citedby_in2.mean.f16.memmap",
    ]
    if meta is not None:
        if not cache_meta_matches(cache_dir, meta):
            return False
    size = n * d * 2
    for name in req:
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) != size:
            return False
    return True


def cache_meta_matches(cache_dir, meta):
    meta_path = cache_paths(cache_dir)["meta"]
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            got = json.load(f)
        return got == meta
    except (OSError, json.JSONDecodeError):
        return False


def edge_split_mask(src_idx, tgt_idx, test_frac, seed):
    n = len(src_idx)
    if test_frac <= 0.0:
        return np.zeros(n, dtype=bool)
    if test_frac >= 1.0:
        return np.ones(n, dtype=bool)
    h = (src_idx.astype(np.int64) * 1000003 + tgt_idx.astype(np.int64) * 9167 + int(seed)) & 0xFFFFFFFF
    thresh = int(test_frac * (2**32 - 1))
    return h < thresh


def load_or_build_degrees(cache_dir, n, id_to_idx, edge_paths, batch_rows, test_frac, split_seed, meta=None):
    paths = cache_paths(cache_dir)
    if os.path.exists(paths["deg_out"]) and os.path.exists(paths["deg_in"]):
        if meta is None or cache_meta_matches(cache_dir, meta):
            deg_out = np.load(paths["deg_out"])
            deg_in = np.load(paths["deg_in"])
            if len(deg_out) == n and len(deg_in) == n:
                return deg_out, deg_in

    deg_out = np.zeros(n, dtype=np.int32)
    deg_in = np.zeros(n, dtype=np.int32)
    for src_ids, tgt_ids in iter_edge_batches(edge_paths, batch_rows=batch_rows, desc="Degrees"):
        src_idx = map_ids(src_ids, id_to_idx)
        tgt_idx = map_ids(tgt_ids, id_to_idx)
        mask = (src_idx >= 0) & (tgt_idx >= 0)
        if not mask.any():
            continue
        src_idx = src_idx[mask]
        tgt_idx = tgt_idx[mask]
        test_mask = edge_split_mask(src_idx, tgt_idx, test_frac, split_seed)
        if test_mask.any():
            keep = ~test_mask
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
        if len(src_idx) == 0:
            continue
        np.add.at(deg_out, src_idx, 1)
        np.add.at(deg_in, tgt_idx, 1)

    np.save(paths["deg_out"], deg_out)
    np.save(paths["deg_in"], deg_in)
    if meta is not None:
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return deg_out, deg_in


def build_query_positives(query_gidx, sub_idx, id_to_idx, edge_paths, batch_rows, direction, test_frac, split_seed):
    qpos = {int(gidx): i for i, gidx in enumerate(query_gidx)}
    qset = np.array(query_gidx, dtype=np.int32)
    pos = [set() for _ in query_gidx]

    for src_ids, tgt_ids in iter_edge_batches(edge_paths, batch_rows=batch_rows, desc="Positives"):
        src_idx = map_ids(src_ids, id_to_idx)
        tgt_idx = map_ids(tgt_ids, id_to_idx)
        mask = (src_idx >= 0) & (tgt_idx >= 0)
        if not mask.any():
            continue
        src_idx = src_idx[mask]
        tgt_idx = tgt_idx[mask]
        if test_frac > 0.0:
            test_mask = edge_split_mask(src_idx, tgt_idx, test_frac, split_seed)
            if not test_mask.any():
                continue
            src_idx = src_idx[test_mask]
            tgt_idx = tgt_idx[test_mask]
        if len(src_idx) == 0:
            continue

        if direction in ("out", "both"):
            m = np.isin(src_idx, qset)
            if m.any():
                ss = src_idx[m]
                tt = tgt_idx[m]
                for s, t in zip(ss, tt):
                    qpos_i = qpos.get(int(s))
                    if qpos_i is None:
                        continue
                    ti = int(sub_idx[t])
                    if ti >= 0 and ti != sub_idx[s]:
                        pos[qpos_i].add(ti)

        if direction in ("in", "both"):
            m = np.isin(tgt_idx, qset)
            if m.any():
                ss = src_idx[m]
                tt = tgt_idx[m]
                for s, t in zip(ss, tt):
                    qpos_i = qpos.get(int(t))
                    if qpos_i is None:
                        continue
                    ti = int(sub_idx[s])
                    if ti >= 0 and ti != sub_idx[t]:
                        pos[qpos_i].add(ti)

    return pos


def build_citation_cache(emb, id_to_idx, edge_paths, cache_dir, batch_rows, vec_chunk, doc_chunk, keep_sums,
                         test_frac, split_seed, meta=None):
    os.makedirs(cache_dir, exist_ok=True)
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
        test_mask = edge_split_mask(src_idx, tgt_idx, test_frac, split_seed)
        if test_mask.any():
            keep = ~test_mask
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
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
    out1_mean = write_means(out1_sum, deg_out, paths["out1_mean"], chunk_docs=doc_chunk)
    in1_mean = write_means(in1_sum, deg_in, paths["in1_mean"], chunk_docs=doc_chunk)
    del out1_mean, in1_mean

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
        test_mask = edge_split_mask(src_idx, tgt_idx, test_frac, split_seed)
        if test_mask.any():
            keep = ~test_mask
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
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
    out2_mean = write_means(out2_sum, cnt_out2, paths["out2_mean"], chunk_docs=doc_chunk)
    in2_mean = write_means(in2_sum, cnt_in2, paths["in2_mean"], chunk_docs=doc_chunk)
    del out2_mean, in2_mean

    np.save(paths["deg_out"], deg_out)
    np.save(paths["deg_in"], deg_in)
    if meta is not None:
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    if not keep_sums:
        del out1_sum, in1_sum, out2_sum, in2_sum
        for k in ["out1_sum", "in1_sum", "out2_sum", "in2_sum"]:
            try:
                os.remove(paths[k])
            except OSError:
                pass


def parse_weights(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = {"self": 1.0, "out1": 0.0, "in1": 0.0, "out2": 0.0, "in2": 0.0}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad weights token: {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"Unknown weight key: {k}")
        out[k] = float(v)
    return out


def parse_vals(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_index(weights, emb, out1, in1, out2, in2, doc_idx, chunk_docs=4096):
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    for s in tqdm(range(0, len(doc_idx), chunk_docs), desc="Build index", unit="doc", dynamic_ncols=True):
        e = min(s + chunk_docs, len(doc_idx))
        idx = doc_idx[s:e]
        vec = weights["self"] * emb[idx].astype(np.float32)
        if weights["out1"] != 0.0:
            vec += weights["out1"] * out1[idx].astype(np.float32)
        if weights["in1"] != 0.0:
            vec += weights["in1"] * in1[idx].astype(np.float32)
        if weights["out2"] != 0.0:
            vec += weights["out2"] * out2[idx].astype(np.float32)
        if weights["in2"] != 0.0:
            vec += weights["in2"] * in2[idx].astype(np.float32)
        faiss.normalize_L2(vec)
        index.add(vec)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--edges", nargs="+", default=[
        os.path.join("openalex_graph_new", "edges_en.parquet"),
        os.path.join("openalex_graph_new", "edges_ru.parquet"),
    ])
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--recompute_cache", action="store_true")
    ap.add_argument("--keep_sums", action="store_true")
    ap.add_argument("--edge_batch_rows", type=int, default=200000)
    ap.add_argument("--vec_chunk", type=int, default=10000)
    ap.add_argument("--doc_chunk", type=int, default=4096)

    ap.add_argument("--eval_mode", choices=["self", "citation"], default="self")
    ap.add_argument("--citation_dir", choices=["out", "in", "both"], default="both")
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--edge_test_frac", type=float, default=0.0)
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--en_parquet", default=os.path.join("openalex_clear_new", "openalex_en.clean.parquet"))
    ap.add_argument("--ru_parquet", default=os.path.join("openalex_clear_new", "openalex_ru.clean.parquet"))
    ap.add_argument("--n_queries", type=int, default=2000)
    ap.add_argument("--ks", type=str, default="1,5,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang_filter", choices=["any", "en", "ru"], default="any")
    ap.add_argument("--q_batch", type=int, default=64)
    ap.add_argument("--q_max_len", type=int, default=96)

    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--self_vals", type=str, default="1.0")
    ap.add_argument("--out1_vals", type=str, default="0.0")
    ap.add_argument("--in1_vals", type=str, default="0.0")
    ap.add_argument("--out2_vals", type=str, default="0.0")
    ap.add_argument("--in2_vals", type=str, default="0.0")
    ap.add_argument("--out_results", default=None)
    args = ap.parse_args()
    if args.edge_test_frac < 0.0 or args.edge_test_frac >= 1.0:
        raise ValueError("--edge_test_frac must be in [0.0, 1.0).")

    emb_path = os.path.join(args.emb_dir, "doc_emb_both.f16.memmap")
    n, d = load_shape(emb_path)
    emb = np.memmap(emb_path, dtype=np.float16, mode="r", shape=(n, d))

    ids_path = os.path.join(args.emb_dir, "doc_ids_both.npy")
    doc_ids = np.load(ids_path, allow_pickle=True)
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        if args.edge_test_frac > 0.0:
            cache_dir = os.path.join(
                args.emb_dir, f"citation_cache_split{args.edge_test_frac}_seed{args.split_seed}"
            )
        else:
            cache_dir = os.path.join(args.emb_dir, "citation_cache")
    meta = {"edge_test_frac": args.edge_test_frac, "split_seed": args.split_seed}
    if args.edge_test_frac > 0.0:
        print(f"Edge split: test_frac={args.edge_test_frac} seed={args.split_seed}")
    if args.recompute_cache or not cache_ready(cache_dir, n, d, meta=meta):
        build_citation_cache(
            emb=emb,
            id_to_idx=id_to_idx,
            edge_paths=args.edges,
            cache_dir=cache_dir,
            batch_rows=args.edge_batch_rows,
            vec_chunk=args.vec_chunk,
            doc_chunk=args.doc_chunk,
            keep_sums=args.keep_sums,
            test_frac=args.edge_test_frac,
            split_seed=args.split_seed,
            meta=meta,
        )
    else:
        print("Cache found:", cache_dir)

    paths = cache_paths(cache_dir)
    out1 = np.memmap(paths["out1_mean"], dtype=np.float16, mode="r", shape=(n, d))
    in1 = np.memmap(paths["in1_mean"], dtype=np.float16, mode="r", shape=(n, d))
    out2 = np.memmap(paths["out2_mean"], dtype=np.float16, mode="r", shape=(n, d))
    in2 = np.memmap(paths["in2_mean"], dtype=np.float16, mode="r", shape=(n, d))

    deg_out, deg_in = load_or_build_degrees(
        cache_dir, n, id_to_idx, args.edges, args.edge_batch_rows, args.edge_test_frac, args.split_seed, meta=meta
    )
    has_cite = (deg_out + deg_in) > 0
    doc_idx = np.flatnonzero(has_cite).astype(np.int32)
    if len(doc_idx) == 0:
        raise RuntimeError("No documents with citations found in the graph.")
    sub_idx = np.full(n, -1, dtype=np.int32)
    sub_idx[doc_idx] = np.arange(len(doc_idx), dtype=np.int32)
    print(f"Docs with citations: {len(doc_idx)}/{n}")

    ks = [int(x) for x in args.ks.split(",")]
    picks = sample_queries(args.en_parquet, args.ru_parquet, args.n_queries, args.seed, args.lang_filter)
    if not picks:
        raise RuntimeError("No queries found. Check parquet paths and language filter.")

    query_gidx = []
    queries = []
    for p in picks:
        idx = id_to_idx.get(p["id"], -1)
        if idx < 0:
            continue
        sub = sub_idx[idx]
        if sub < 0:
            continue
        query_gidx.append(int(idx))
        queries.append(p["q"])

    if args.eval_mode == "citation":
        pos_sets = build_query_positives(
            query_gidx, sub_idx, id_to_idx, args.edges,
            args.edge_batch_rows, args.citation_dir, args.edge_test_frac, args.split_seed
        )
        filtered_gidx = []
        filtered_queries = []
        filtered_pos = []
        for gidx, q, pos in zip(query_gidx, queries, pos_sets):
            if not pos:
                continue
            filtered_gidx.append(gidx)
            filtered_queries.append(q)
            filtered_pos.append(pos)
        query_gidx = filtered_gidx
        queries = filtered_queries
        pos_sets = filtered_pos
        print(f"Queries kept: {len(queries)}/{len(picks)} (with citations)")
        if not queries:
            raise RuntimeError("No queries with citation positives in the subset.")
    else:
        gt_idx = [int(sub_idx[g]) for g in query_gidx]
        print(f"Queries kept: {len(queries)}/{len(picks)}")
        if not queries:
            raise RuntimeError("No queries with matching ids in the cited/citing subset.")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    mdl = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto").eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    qvec = embed_queries(tok, mdl, queries, bs=args.q_batch, max_len=args.q_max_len)

    if args.weights:
        weight_sets = [parse_weights(s) for s in args.weights]
    else:
        weight_sets = []
        for vals in itertools.product(
            parse_vals(args.self_vals),
            parse_vals(args.out1_vals),
            parse_vals(args.in1_vals),
            parse_vals(args.out2_vals),
            parse_vals(args.in2_vals),
        ):
            weight_sets.append({"self": vals[0], "out1": vals[1], "in1": vals[2], "out2": vals[3], "in2": vals[4]})

    if args.out_results:
        os.makedirs(os.path.dirname(args.out_results) or ".", exist_ok=True)

    for w in weight_sets:
        tag = f"self={w['self']},out1={w['out1']},in1={w['in1']},out2={w['out2']},in2={w['in2']}"
        print(f"\n=== Weights: {tag} ===")
        index = build_index(w, emb, out1, in1, out2, in2, doc_idx, chunk_docs=args.doc_chunk)
        _, I = index.search(qvec, max(ks))
        if args.eval_mode == "citation":
            exclude_idx = None
            if args.exclude_self:
                exclude_idx = [int(sub_idx[g]) for g in query_gidx]
            res = metrics_multi(pos_sets, I, ks, exclude_idx=exclude_idx)
        else:
            res = metrics_idx(gt_idx, I, ks)
        for k in ks:
            print(f"P@{k}={res[f'P@{k}']:.4f}  R@{k}={res[f'R@{k}']:.4f}  MRR@{k}={res[f'MRR@{k}']:.4f}")
        if args.out_results:
            row = {"self": w["self"], "out1": w["out1"], "in1": w["in1"], "out2": w["out2"], "in2": w["in2"]}
            row.update(res)
            row["eval_mode"] = args.eval_mode
            row["citation_dir"] = args.citation_dir if args.eval_mode == "citation" else None
            row["queries"] = len(queries)
            row["docs"] = int(len(doc_idx))
            row["edge_test_frac"] = args.edge_test_frac
            row["split_seed"] = args.split_seed
            with open(args.out_results, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
