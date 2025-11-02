#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np, faiss
from tqdm import tqdm

def load_memmap(mem_path):
    with open(mem_path+".shape.json","r",encoding="utf-8") as f:
        n, d = json.load(f)
    arr = np.memmap(mem_path, dtype=np.float16, mode="r", shape=(n,d))
    return arr, n, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--memfile", default="doc_emb_both.f16.memmap")
    ap.add_argument("--out", default=None, help="путь к .index, по умолчанию <emb_dir>/faiss_both.index")
    ap.add_argument("--index_type", choices=["flat","ivfpq"], default="ivfpq")
    ap.add_argument("--metric", choices=["ip","l2"], default="ip")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--train_size", type=int, default=500000)
    args = ap.parse_args()

    mem_path = os.path.join(args.emb_dir, args.memfile)
    Xh, N, D = load_memmap(mem_path)
    out = args.out or os.path.join(args.emb_dir, "faiss_both.index")
    metric = faiss.METRIC_INNER_PRODUCT if args.metric=="ip" else faiss.METRIC_L2

    # загрузим выборку для тренировки
    take = min(args.train_size, N)
    idx = np.random.RandomState(0).choice(N, size=take, replace=False)
    Xtrain = np.array(Xh[idx], dtype=np.float32)
    Xall = np.array(Xh, dtype=np.float32)

    if args.index_type=="flat":
        index = faiss.IndexFlatIP(D) if args.metric=="ip" else faiss.IndexFlatL2(D)
    else:
        descr = f"IVF{args.nlist},PQ{args.m}x{args.nbits}"
        index = faiss.index_factory(D, descr, metric)
        index.train(Xtrain)

    # добавляем
    for s in tqdm(range(0, N, 200000), desc="Adding", unit="vecs", dynamic_ncols=True):
        e = min(s+200000, N)
        index.add(Xall[s:e])

    faiss.write_index(index, out)
    meta = {"N": N, "D": D, "metric": args.metric, "type": args.index_type,
            "nlist": getattr(index, "nlist", None)}
    with open(out+".meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved:", out)

if __name__ == "__main__":
    main()
