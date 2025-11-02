#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np, polars as pl, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

MODEL_ID = "intfloat/multilingual-e5-large"

def mean_pool(last_hidden_state, attention_mask):
    m = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)

@torch.inference_mode()
def embed_texts(tok, mdl, texts, max_len=512, batch=32, prefix="passage: "):
    if not texts: return np.zeros((0, 1024), dtype=np.float16)
    out=[]
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", unit="batch", dynamic_ncols=True):
        b = [prefix + t for t in texts[i:i+batch]]
        enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k:v.to(mdl.device) for k,v in enc.items()}
        h = mdl(**enc).last_hidden_state
        v = mean_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1).to(torch.float16).cpu().numpy()
        out.append(v)
    return np.vstack(out)

def read_lang(parquet, lang, limit=0):
    if not parquet: return [], []
    df = (pl.read_parquet(parquet, use_pyarrow=True)
            .select("id","language","title","abstract_text")
            .filter(pl.col("language")==lang)
            .with_columns([ (pl.col("title").fill_null("") + pl.lit("\n\n") +
                             pl.col("abstract_text").fill_null("")).alias("txt") ]))
    if limit>0: df = df.head(limit)
    return df["id"].to_list(), df["txt"].to_list()

def save_memmap(path, arr, chunk=8192):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fp = np.memmap(path, dtype=np.float16, mode="w+", shape=arr.shape)
    for s in tqdm(range(0, arr.shape[0], chunk), desc="Saving", unit="chunk", dynamic_ncols=True):
        fp[s:s+chunk] = arr[s:s+chunk]
    del fp
    with open(path+".shape.json","w",encoding="utf-8") as f:
        json.dump([int(arr.shape[0]), int(arr.shape[1])], f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en_parquet")
    ap.add_argument("--ru_parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--doc_max_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit_en", type=int, default=0)
    ap.add_argument("--limit_ru", type=int, default=0)
    ap.add_argument("--lora_dir", help="опционально: путь к LoRA-адаптеру для пересчёта эмбеддингов")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    base = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()
    mdl = PeftModel.from_pretrained(base, args.lora_dir).eval() if args.lora_dir else base
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    en_ids, en_txt = read_lang(args.en_parquet, "en", args.limit_en)
    ru_ids, ru_txt = read_lang(args.ru_parquet, "ru", args.limit_ru)

    if en_ids:
        en_emb = embed_texts(tok, mdl, en_txt, args.doc_max_len, args.batch, "passage: ")
        np.save(os.path.join(args.outdir,"doc_ids_en.npy"), np.array(en_ids, dtype=object))
        save_memmap(os.path.join(args.outdir,"doc_emb_en.f16.memmap"), en_emb)

    if ru_ids:
        ru_emb = embed_texts(tok, mdl, ru_txt, args.doc_max_len, args.batch, "passage: ")
        np.save(os.path.join(args.outdir,"doc_ids_ru.npy"), np.array(ru_ids, dtype=object))
        save_memmap(os.path.join(args.outdir,"doc_emb_ru.f16.memmap"), ru_emb)

    # both
    ids = (en_ids or []) + (ru_ids or [])
    if ids:
        emb = []
        if en_ids: emb.append(en_emb)
        if ru_ids: emb.append(ru_emb)
        both = np.vstack(emb)
        save_memmap(os.path.join(args.outdir,"doc_emb_both.f16.memmap"), both)
        np.save(os.path.join(args.outdir,"doc_ids_both.npy"), np.array(ids, dtype=object))
        langs = (["en"]*len(en_ids)) + (["ru"]*len(ru_ids))
        np.save(os.path.join(args.outdir,"doc_lang_both.npy"), np.array(langs, dtype="<U2"))

if __name__ == "__main__":
    main()
