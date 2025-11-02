#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, re
import numpy as np, polars as pl, faiss, torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

MODEL_ID = "intfloat/multilingual-e5-large"

def mean_pool(h, m):
    m = m.unsqueeze(-1).expand(h.size()).float()
    return (h*m).sum(1)/torch.clamp(m.sum(1), min=1e-9)

@torch.inference_mode()
def embed_queries(tok, mdl, texts, max_len=96, bs=64):
    if not texts: return np.zeros((0,1), dtype="float32")
    out=[]
    for i in range(0, len(texts), bs):
        batch = ["query: "+t for t in texts[i:i+bs]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k:v.to(mdl.device) for k,v in enc.items()}
        h = mdl(**enc).last_hidden_state
        v = mean_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1).cpu().numpy().astype("float32")
        out.append(v)
    return np.vstack(out)

@torch.inference_mode()
def embed_passages(tok, mdl, texts, max_len=512, bs=64):
    if not texts: return np.zeros((0,1), dtype="float32")
    out=[]
    for i in range(0, len(texts), bs):
        batch = ["passage: "+t for t in texts[i:i+bs]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k:v.to(mdl.device) for k,v in enc.items()}
        h = mdl(**enc).last_hidden_state
        v = mean_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1).cpu().numpy().astype("float32")
        out.append(v)
    return np.vstack(out)

def load_model(lora_dir=None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    base = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto").eval()
    if lora_dir:
        base = PeftModel.from_pretrained(base, lora_dir).eval()
    # TF32 опционален
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, base

def load_index(emb_dir, index_name, nprobe):
    index = faiss.read_index(os.path.join(emb_dir, index_name))
    try: index.nprobe = nprobe
    except Exception: pass
    ids = np.load(os.path.join(emb_dir,"doc_ids_both.npy"), allow_pickle=True)
    langs = None
    p_lang = os.path.join(emb_dir,"doc_lang_both.npy")
    if os.path.exists(p_lang):
        langs = np.load(p_lang, allow_pickle=True)
    return index, ids, langs

def normalize_openalex_id(s: str) -> str:
    s = str(s)
    return re.sub(r"^https?://(www\.)?openalex\.org/", "", s)

def openalex_url(did: str) -> str:
    s = str(did)
    return s if s.startswith("http") else f"https://openalex.org/{s}"

def build_lookup(en_parquet, ru_parquet):
    if not en_parquet and not ru_parquet:
        return {}
    cols = ["id","language","title","abstract_text","publication_year","cited_by_count"]
    dfs=[]
    if en_parquet and os.path.exists(en_parquet):
        df_en = pl.read_parquet(en_parquet, use_pyarrow=True)
        dfs.append(df_en.select([c for c in cols if c in df_en.columns]))
    if ru_parquet and os.path.exists(ru_parquet):
        df_ru = pl.read_parquet(ru_parquet, use_pyarrow=True)
        dfs.append(df_ru.select([c for c in cols if c in df_ru.columns]))
    if not dfs: return {}
    df = pl.concat(dfs, how="vertical_relaxed")
    lut = {}
    for r in df.iter_rows(named=True):
        rid = r["id"]
        rid_norm = normalize_openalex_id(rid)
        meta = {
            "lang": r.get("language"),
            "title": r.get("title") or "",
            "abstract": r.get("abstract_text") or "",
            "year": r.get("publication_year"),
            "cited_by": r.get("cited_by_count", 0)
        }
        # кладём по двум ключам: как есть и нормализованный
        lut[str(rid)] = meta
        lut[rid_norm] = meta
    return lut

def doc_text(meta, mode="title+abstract"):
    if not meta: return ""
    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    if mode == "abstract_only":
        return abstract
    return ((title + "\n\n") if title else "") + abstract

def shorten(s, n):
    s = (s or "").replace("\n", " ").strip()
    return (s[:n-1] + "…") if n and len(s) > n else s

def fmt_info(meta):
    if not meta: return ""
    lang = meta.get("lang") or "??"
    year = meta.get("year")
    cites = meta.get("cited_by")
    parts=[lang]
    if year is not None: parts.append(str(year))
    if cites is not None: parts.append(f"cites={cites}")
    return " | ".join(parts)

def main():
    ap = argparse.ArgumentParser(description="Interactive search tester for E5 (LoRA-ready).")
    ap.add_argument("--emb_dir", required=True, help="Папка с memmap/ids и FAISS-индексом")
    ap.add_argument("--index_name", default="faiss_both.index")
    ap.add_argument("--en_parquet", help="Parquet для EN (метаданные/тексты)")
    ap.add_argument("--ru_parquet", help="Parquet для RU (метаданные/тексты)")
    ap.add_argument("--lora_dir", help="Путь к LoRA-адаптеру (для кодирования запросов)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--topk_fetch", type=int, default=200)
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--lang_filter", choices=["any","en","ru"], default="any")
    ap.add_argument("--q_max", type=int, default=96)
    ap.add_argument("--rerank", action="store_true", help="Пересчитать топ-N кандидатов по тексту")
    ap.add_argument("--rerank_max_len", type=int, default=512)
    ap.add_argument("--doc_text_source", choices=["title+abstract","abstract_only"], default="title+abstract")
    ap.add_argument("--show_abstract_chars", type=int, default=0, help="Показать усечённый abstract (0=не показывать)")
    args = ap.parse_args()

    tok, mdl = load_model(args.lora_dir)
    index, doc_ids, doc_langs = load_index(args.emb_dir, args.index_name, args.nprobe)
    lut = build_lookup(args.en_parquet, args.ru_parquet)

    print("Введите запросы построчно (пустая строка — запуск поиска):")
    buf=[]
    while True:
        try:
            line = input().rstrip("\n")
        except EOFError:
            break
        if line=="":
            if not buf:
                print("Нет запросов. Введите хотя бы один.")
                continue
            # embed queries
            qvec = embed_queries(tok, mdl, buf, max_len=args.q_max, bs=64)
            norms = np.linalg.norm(qvec, axis=1)
            print(f"\ndebug L2 norms: {np.round(norms,3).tolist()}")
            # FAISS search
            D, I = index.search(qvec, args.topk_fetch)
            for qi, q in enumerate(buf):
                print(f"\nQ: {q}")
                cand = []
                limit = args.topk_fetch if args.rerank else args.topk  # FIX: при rerank берём topk_fetch
                for j, row in enumerate(I[qi].tolist()):
                    did_raw = doc_ids[row]
                    # язык
                    if args.lang_filter != "any":
                        lang_ok = True
                        if doc_langs is not None:
                            lang_ok = (doc_langs[row] == args.lang_filter)
                        else:
                            meta = lut.get(str(did_raw)) or lut.get(normalize_openalex_id(did_raw))
                            lang_ok = (meta and meta.get("lang")==args.lang_filter)
                        if not lang_ok:
                            continue
                    cand.append((did_raw, float(D[qi][j]), row))
                    if len(cand) >= limit:
                        break  # FIX: раньше не было break
                # optional rerank
                if args.rerank and cand:
                    ids = [c[0] for c in cand]
                    metas = [lut.get(str(x)) or lut.get(normalize_openalex_id(x)) for x in ids]
                    texts = [doc_text(m, args.doc_text_source) if m else "" for m in metas]
                    mask = [t.strip()!="" for t in texts]
                    emb = np.zeros((len(texts), qvec.shape[1]), dtype="float32")
                    if any(mask):
                        emb[mask] = embed_passages(tok, mdl, [t if m else "" for t,m in zip(texts,mask)],
                                                   max_len=args.rerank_max_len, bs=64)
                    sims = (qvec[qi:qi+1] @ emb.T).ravel()
                    cand = [(ids[i], float(sims[i] if mask[i] else cand[i][1]), cand[i][2]) for i in range(len(ids))]
                    cand.sort(key=lambda x: x[1], reverse=True)
                # print topk
                for rank, (did_raw, sim, row) in enumerate(cand[:args.topk], start=1):
                    url = openalex_url(normalize_openalex_id(did_raw))
                    meta = lut.get(str(did_raw)) or lut.get(normalize_openalex_id(did_raw)) or {}
                    title = (meta.get("title") or "").strip() or "(no title)"
                    info = fmt_info(meta)
                    print(f"  {rank}. {url}  sim={sim:.4f}  [{info}]")
                    print(f"     {shorten(title, 180)}")
                    if args.show_abstract_chars > 0:
                        abs_short = shorten(meta.get("abstract",""), args.show_abstract_chars)
                        if abs_short:
                            print(f"     └ {abs_short}")
            buf.clear()
            print("\nВведите новые запросы (пустая строка — поиск, Ctrl+C — выход):")
            continue
        buf.append(line)

if __name__ == "__main__":
    main()
