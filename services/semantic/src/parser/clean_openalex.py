import argparse, os, sys, gzip, datetime
import polars as pl

# ---------- helpers (только векторные операции) ----------

def norm_whitespace(expr: pl.Expr) -> pl.Expr:
    # схлопываем пробелы и обрезаем края (совместимо со старыми Polars)
    return expr.fill_null("").cast(pl.Utf8).str.replace_all(r"\s+", " ").str.strip_chars()

def norm_text(expr: pl.Expr) -> pl.Expr:
    # нижний регистр + схлопывание пробелов + обрезка краёв
    return expr.fill_null("").cast(pl.Utf8).str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()

def norm_doi(expr: pl.Expr) -> pl.Expr:
    # приводим DOI к канону: нижний регистр, без https://doi.org/ и префикса doi:
    return (
        expr.fill_null("")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all(r"^https?://(dx\.)?doi\.org/", "")
            .str.replace_all(r"^doi:", "")
            .str.strip_chars()
    )

def write_outputs(df: pl.DataFrame, out_parquet: str, out_csv_gz: str | None):
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    df.write_parquet(out_parquet, compression="zstd", statistics=True)
    if out_csv_gz:
        with gzip.open(out_csv_gz, "wt", encoding="utf-8", newline="") as f:
            df.write_csv(f)

# ---------- pipeline ----------

def build_pipeline(
    glob_pattern: str,
    expected_lang: str,
    min_title_chars: int,
    min_abs_chars: int,
    year_min: int,
    year_max: int,
) -> pl.LazyFrame:

    scan = pl.scan_csv(
        glob_pattern,
        has_header=True,
        ignore_errors=True,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )

    # базовая типизация часто используемых колонок
    df = (
        scan
        .with_columns([
            pl.col("title").cast(pl.Utf8),
            pl.col("abstract_text").cast(pl.Utf8),
            pl.col("language").cast(pl.Utf8),
            pl.col("publication_year").cast(pl.Int64),
            pl.col("cited_by_count").cast(pl.Int64).fill_null(0),
        ])
        # нормализованные поля
        .with_columns([
            norm_text(pl.col("title")).alias("title_norm"),
            norm_text(pl.col("abstract_text")).alias("abstract_norm"),
            norm_doi(pl.col("doi")).alias("doi_norm"),
        ])
        # длины
        .with_columns([
            pl.col("title_norm").str.len_chars().alias("title_len"),
            pl.col("abstract_norm").str.len_chars().alias("abstract_len"),
        ])
        # фильтры качества
        .filter(
            (pl.col("title_len") >= min_title_chars) &
            (pl.col("abstract_len") >= min_abs_chars) &
            (pl.col("publication_year") >= year_min) &
            (pl.col("publication_year") <= year_max) &
            (pl.col("language") == expected_lang)
        )
        # признаки "качества" для выбора при дедупе
        .with_columns([
            pl.col("cited_by_count").fill_null(0).alias("_q_citations"),
            pl.col("abstract_len").alias("_q_abslen"),
            pl.col("publication_year").fill_null(0).alias("_q_year"),
            (pl.col("doi_norm") != "").alias("_has_doi"),
        ])
    )

    # 1) дедуп по DOI: сначала записи с DOI, сортируем по качеству, берём первые
    df_sorted = df.sort(
        ["_has_doi", "_q_citations", "_q_abslen", "_q_year"],
        descending=[True, True, True, True],
        nulls_last=True,
    )

    df_dedup_doi = pl.concat([
        df_sorted.filter(pl.col("_has_doi")).unique(subset=["doi_norm"], keep="first"),
        df_sorted.filter(~pl.col("_has_doi")),
    ])

    # 2) дедуп по id (подстраховка)
    df_dedup_id = df_dedup_doi.sort(
        ["_q_citations", "_q_abslen", "_q_year"],
        descending=[True, True, True],
        nulls_last=True,
    ).unique(subset=["id"], keep="first")

    # 3) дедуп по контенту (title_norm + abstract_norm) через хеш
    df_fp = (
        df_dedup_id
        .with_columns([
            pl.concat_str([pl.col("title_norm"), pl.lit("||"), pl.col("abstract_norm")], separator="")
              .hash(seed=42).alias("_fp")
        ])
        .sort(
            ["_q_citations", "_q_abslen", "_q_year"],
            descending=[True, True, True],
            nulls_last=True,
        )
        .unique(subset=["_fp"], keep="first")
        .drop(["_fp"])
    )

    # убрать служебные колонки, оставить рабочие
    cleaned = df_fp.drop([
        "title_norm", "abstract_norm", "doi_norm",
        "title_len", "abstract_len",
        "_q_citations", "_q_abslen", "_q_year", "_has_doi",
    ])

    return cleaned

# ---------- CLI ----------

def main():
    import datetime as _dt
    ap = argparse.ArgumentParser(description="Clean OpenAlex CSV chunks: drop empties & dedup by DOI/ID/content fingerprint.")
    ap.add_argument("--indir", required=True, help="Папка с чанками CSV(.gz)")
    ap.add_argument("--outdir", required=True, help="Куда писать очищенные файлы")
    ap.add_argument("--min-title-chars", type=int, default=5)
    ap.add_argument("--min-abstract-chars", type=int, default=30)
    ap.add_argument("--year-min", type=int, default=1900)
    ap.add_argument("--year-max", type=int, default=_dt.datetime.now().year)
    ap.add_argument("--write-csv", action="store_true", help="Помимо Parquet, сохранить и CSV.GZ")
    args = ap.parse_args()

    patterns = {
        "en": os.path.join(args.indir, "openalex_en.part*.csv*"),
        "ru": os.path.join(args.indir, "openalex_ru.part*.csv*"),
    }

    os.makedirs(args.outdir, exist_ok=True)

    for lang, pat in patterns.items():
        print(f"[{lang}] scan: {pat}", file=sys.stderr)
        df_clean_lazy = build_pipeline(
            pat, lang,
            args.min_title_chars, args.min_abstract_chars,
            args.year_min, args.year_max
        )
        df_clean = df_clean_lazy.collect()
        n_out = df_clean.height
        print(f"[{lang}] cleaned rows: {n_out}", file=sys.stderr)

        out_parquet = os.path.join(args.outdir, f"openalex_{lang}.clean.parquet")
        out_csv_gz = os.path.join(args.outdir, f"openalex_{lang}.clean.csv.gz") if args.write_csv else None
        write_outputs(df_clean, out_parquet, out_csv_gz)
        print(f"[{lang}] DONE → {out_parquet}" + (f" & {out_csv_gz}" if out_csv_gz else ""), file=sys.stderr)

if __name__ == "__main__":
    main()
