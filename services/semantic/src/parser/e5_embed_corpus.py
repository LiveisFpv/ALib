#!/usr/bin/env python3
"""Generate document embeddings from Postgres data for FAISS indexing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import psycopg
import torch
from psycopg.rows import dict_row
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:  # optional LoRA support
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None

from psycopg import Connection

from src.config.config import DATABASE_SETTINGS, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME

EMBEDDING_DIM = 1024


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.inference_mode()
def embed_texts(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    *,
    max_len: int,
    batch: int,
    prefix: str = "passage: ",
) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float16)

    outputs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch), desc="Embedding", unit="batch", dynamic_ncols=True):
        batch_texts = [prefix + t for t in texts[start : start + batch]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        pooled = mean_pool(hidden, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.to(torch.float16).cpu().numpy())

    return np.vstack(outputs)


def prepare_model(model_id: str, lora_dir: str | None) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if lora_dir and PeftModel is not None:
        base_model = PeftModel.from_pretrained(base_model, lora_dir).to(device).eval()
    return tokenizer, base_model


def count_documents(conn: Connection, where_clause: str | None) -> int:
    query = "SELECT COUNT(*) AS cnt FROM papers WHERE (COALESCE(title, '') <> '' OR COALESCE(abstract, '') <> '')"
    if where_clause:
        query += f" AND ({where_clause})"
    with conn.cursor() as cur:
        cur.execute(query)
        return int(cur.fetchone()["cnt"])


def stream_documents(
    conn: Connection,
    *,
    batch_size: int,
    where_clause: str | None,
    limit: int | None,
) -> Iterable[list[dict]]:
    base_query = (
        "SELECT paper_id, COALESCE(title, '') AS title, COALESCE(abstract, '') AS abstract "
        "FROM papers "
        "WHERE (COALESCE(title, '') <> '' OR COALESCE(abstract, '') <> '')"
    )
    if where_clause:
        base_query += f" AND ({where_clause})"
    base_query += " ORDER BY paper_id"
    if limit is not None:
        base_query += f" LIMIT {int(limit)}"

    with conn.cursor(name="papers_stream") as cur:
        cur.itersize = batch_size
        cur.execute(base_query)
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            yield [dict(row) for row in batch]


def format_document(row: dict) -> tuple[int, str]:
    text_parts = []
    if row.get("title"):
        text_parts.append(str(row["title"]).strip())
    if row.get("abstract"):
        text_parts.append(str(row["abstract"]).strip())
    combined = "\n\n".join(part for part in text_parts if part)
    return int(row["paper_id"]), combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed documents from Postgres for FAISS")
    parser.add_argument("--outdir", required=True, help="Directory for embedding outputs")
    parser.add_argument("--db-url", help="Optional explicit Postgres URL")
    parser.add_argument("--where", help="Additional SQL filter for documents")
    parser.add_argument("--limit", type=int, help="Limit number of documents")
    parser.add_argument("--doc-max-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--lora-dir", help="Path to LoRA adapter (optional)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = prepare_model(args.model, args.lora_dir)

    db_dsn = args.db_url or DATABASE_SETTINGS.psycopg_dsn()
    conn = psycopg.connect(db_dsn, row_factory=dict_row)
    try:
        total_docs = count_documents(conn, args.where)
        if args.limit is not None:
            total_docs = min(total_docs, args.limit)
        if total_docs == 0:
            raise SystemExit("No documents found to embed")

        embeddings_path = outdir / "doc_embeddings.f16.memmap"
        ids_path = outdir / "doc_ids.npy"
        meta_path = embeddings_path.with_suffix(".shape.json")

        memmap = np.memmap(embeddings_path, dtype=np.float16, mode="w+", shape=(total_docs, EMBEDDING_DIM))
        ids = np.empty(total_docs, dtype=np.int64)

        offset = 0
        for batch in stream_documents(conn, batch_size=args.batch, where_clause=args.where, limit=args.limit):
            doc_ids: list[int] = []
            texts: list[str] = []
            for row in batch:
                doc_id, combined = format_document(row)
                if not combined:
                    continue
                doc_ids.append(doc_id)
                texts.append(combined)

            if not doc_ids:
                continue

            embeddings = embed_texts(
                tokenizer,
                model,
                texts,
                max_len=args.doc_max_len,
                batch=args.batch,
                prefix="passage: ",
            )

            batch_size = embeddings.shape[0]
            memmap[offset : offset + batch_size] = embeddings
            ids[offset : offset + batch_size] = np.array(doc_ids, dtype=np.int64)
            offset += batch_size

        actual_docs = offset
        memmap.flush()
        del memmap

        bytes_needed = actual_docs * EMBEDDING_DIM * np.dtype(np.float16).itemsize
        os.truncate(embeddings_path, bytes_needed)
        ids = ids[:actual_docs]
        np.save(ids_path, ids)
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump([int(ids.shape[0]), EMBEDDING_DIM], meta_file)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
