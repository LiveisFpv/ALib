import os, polars as pl, argparse

def explode_edges(parquet_path: str, out_path: str):
    df = pl.read_parquet(parquet_path, use_pyarrow=True)
    if "referenced_works" not in df.columns:
        raise SystemExit(f"no referenced_works in {parquet_path}")
    # split по " | " -> explode -> убрать пустые
    edges = (df.select([
                pl.col("id").alias("source_id"),
                pl.col("referenced_works").str.split(" | ")
            ])
            .explode("referenced_works")
            .with_columns([
                pl.col("referenced_works").str.strip_chars().alias("target_id")
            ])
            .filter(pl.col("target_id") != "")
            .unique()
           )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    edges.write_parquet(out_path, compression="zstd")
    print(f"edges: {parquet_path} -> {out_path}  rows={edges.height}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    explode_edges(os.path.join(args.indir, "openalex_en.clean.parquet"),
                  os.path.join(args.outdir, "edges_en.parquet"))
    explode_edges(os.path.join(args.indir, "openalex_ru.clean.parquet"),
                  os.path.join(args.outdir, "edges_ru.parquet"))
