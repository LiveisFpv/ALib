import argparse, csv, gzip, json, os, sys, time
from typing import Any, Dict, List, Optional
import requests

API = "https://api.openalex.org/works"

# ---------- utils ----------

def reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not inv_idx or not isinstance(inv_idx, dict):
        return None
    max_pos = -1
    for positions in inv_idx.values():
        if positions:
            mp = max(positions)
            if mp > max_pos:
                max_pos = mp
    if max_pos < 0:
        return None
    toks = [""] * (max_pos + 1)
    for token, positions in inv_idx.items():
        for p in positions:
            if 0 <= p < len(toks):
                toks[p] = token
    text = " ".join(toks).strip()
    return " ".join(text.split()) or None

def list_join(values: Optional[List[Any]]) -> str:
    if not values: return ""
    return " | ".join(str(v) for v in values)

def polite_sleep(last_req_ts: List[float], rps: float = 8.0):
    min_interval = 1.0 / rps
    now = time.time()
    if last_req_ts and now - last_req_ts[0] < min_interval:
        time.sleep(min_interval - (now - last_req_ts[0]))
    last_req_ts[:] = [time.time()]

def load_state(p: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def save_state(p: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def open_csv_writer(path: str, fieldnames: List[str], gzip_flag: bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gzip_flag:
        f = gzip.open(path, "wt", encoding="utf-8", newline="")
    else:
        f = open(path, "w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    return f, w

def make_chunk_path(outdir: str, lang: str, idx: int, gz: bool) -> str:
    base = f"openalex_{lang}.part{idx:03d}.csv"
    return os.path.join(outdir, base + (".gz" if gz else ""))

# ---------- SELECT (только нужные поля) ----------

SELECT_FIELDS = ",".join([
    "id","doi","title","display_name","ids","language",
    "publication_year","publication_date","type",
    "cited_by_count",
    "abstract_inverted_index",
    "referenced_works","related_works","referenced_works_count",
    "topics","concepts","keywords",
    "primary_location","best_oa_location",
    "authorships",
    "open_access"
])

def build_params(lang: str, email: str, cursor: str) -> Dict[str, Any]:
    return {
        "filter": f"language:{lang},type:article,is_paratext:false,is_retracted:false",
        "sort": "cited_by_count:desc",
        "per-page": 200,
        "cursor": cursor,
        "select": SELECT_FIELDS,
        "mailto": email,
    }

# ---------- FLATTEN (минимальный и компактный) ----------

def flatten_work(w: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "id": w.get("id",""),
        "doi": w.get("doi","") or (w.get("ids") or {}).get("doi",""),
        "title": w.get("title","") or w.get("display_name",""),
        "language": w.get("language",""),
        "publication_year": w.get("publication_year",""),
        "publication_date": w.get("publication_date",""),
        "type": w.get("type",""),
        "cited_by_count": w.get("cited_by_count",""),
        "abstract_text": reconstruct_abstract(w.get("abstract_inverted_index")) or "",
        # граф цитирования
        "referenced_works_count": w.get("referenced_works_count",""),
        "referenced_works": list_join(w.get("referenced_works")),
        "related_works": list_join(w.get("related_works")),
    }

    # темы/ключевые/концепты — только имена и id (без громоздких JSON)
    topics = w.get("topics") or []
    concepts = w.get("concepts") or []
    keywords = w.get("keywords") or []
    out.update({
        "topics_ids": list_join([t.get("id","") for t in topics if t.get("id")]),
        "topics_names": list_join([t.get("display_name","") for t in topics if t.get("display_name")]),
        "concepts_ids": list_join([c.get("id","") for c in concepts if c.get("id")]),
        "concepts_names": list_join([c.get("display_name","") for c in concepts if c.get("display_name")]),
        "keywords_names": list_join([k.get("display_name","") for k in keywords if k.get("display_name")]),
    })

    # упрощённые primary/best_oa — только полезные для ссылки/фильтра
    def flat_loc(prefix: str, loc: Optional[Dict[str, Any]]):
        if not loc:
            return {
                f"{prefix}_is_oa": "",
                f"{prefix}_landing_page_url": "",
                f"{prefix}_pdf_url": "",
                f"{prefix}_license": "",
                f"{prefix}_source_id": "",
                f"{prefix}_source_display_name": "",
                f"{prefix}_source_type": "",
                f"{prefix}_source_issn_l": ""
            }
        src = loc.get("source") or {}
        return {
            f"{prefix}_is_oa": loc.get("is_oa",""),
            f"{prefix}_landing_page_url": loc.get("landing_page_url",""),
            f"{prefix}_pdf_url": loc.get("pdf_url",""),
            f"{prefix}_license": loc.get("license",""),
            f"{prefix}_source_id": src.get("id",""),
            f"{prefix}_source_display_name": src.get("display_name",""),
            f"{prefix}_source_type": src.get("type",""),
            f"{prefix}_source_issn_l": src.get("issn_l",""),
        }

    out.update(flat_loc("primary_loc", w.get("primary_location")))
    out.update(flat_loc("best_oa_loc", w.get("best_oa_location")))

    # авторы/аффилиации — компактно (только имена/ID/ORCID и имена организаций/коды стран)
    aus = w.get("authorships") or []
    authors_ids, authors_names, authors_orcids = [], [], []
    inst_names, inst_cc = [], []
    for a in aus:
        author = a.get("author") or {}
        if author.get("id"): authors_ids.append(author["id"])
        if author.get("display_name"): authors_names.append(author["display_name"])
        if author.get("orcid"): authors_orcids.append(author["orcid"])
        for inst in a.get("institutions") or []:
            if inst.get("display_name"): inst_names.append(inst["display_name"])
            if inst.get("country_code"): inst_cc.append(inst["country_code"])
    out.update({
        "authors_ids": list_join(authors_ids),
        "authors_names": list_join(authors_names),
        "authors_orcids": list_join(authors_orcids),
        "institutions_names": list_join(inst_names),
        "institutions_country_codes": list_join(inst_cc),
    })

    # open access (кратко)
    oa = w.get("open_access") or {}
    out.update({
        "oa_is_oa": oa.get("is_oa",""),
        "oa_status": oa.get("oa_status",""),
        "oa_url": oa.get("oa_url",""),
    })

    return out

# ---------- fetch (резюмируемо, чанки, gzip) ----------

def stream_lang(lang: str, target_n: int, email: str, outdir: str,
                chunk_size: int, gzip_flag: bool, resume: bool):
    sess = requests.Session()
    headers = {"User-Agent": f"OpenAlexCsvLean/1.0 (+mailto:{email})"}

    state_dir = os.path.join(outdir, "state")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"state_{lang}.json")

    cursor = "*"; collected = 0; chunk_idx = 1
    if resume:
        st = load_state(state_path)
        if st:
            cursor = st.get("cursor","*") or "*"
            collected = int(st.get("collected",0))
            chunk_idx = int(st.get("chunk_idx",1))
            print(f"[{lang}] resume: collected={collected}, chunk={chunk_idx}, cursor={cursor}", file=sys.stderr)

    fcsv = None; writer = None; in_chunk = 0
    last_req_ts = [0.0]

    while collected < target_n and cursor:
        params = {
            "filter": f"language:{lang},type:article,is_paratext:false,is_retracted:false",
            "sort": "cited_by_count:desc",
            "per-page": 200,
            "cursor": cursor,
            "select": SELECT_FIELDS,
            "mailto": email,
        }
        polite_sleep(last_req_ts, rps=8.0)
        resp = sess.get(API, params=params, headers=headers, timeout=60)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After","2"))
            print(f"[{lang}] 429 — sleep {wait}s", file=sys.stderr)
            time.sleep(max(2, wait)); continue
        if resp.status_code >= 500:
            print(f"[{lang}] {resp.status_code} — server error, retry in 5s", file=sys.stderr)
            time.sleep(5); continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"[{lang}] HTTP error: {e}  body={resp.text[:300]}", file=sys.stderr)
            break

        data = resp.json()
        results = data.get("results", [])
        cursor = data.get("meta", {}).get("next_cursor")

        if not results:
            break

        for w in results:
            row = flatten_work(w)
            if writer is None:
                path = make_chunk_path(outdir, lang, chunk_idx, gzip_flag)
                fcsv, writer = open_csv_writer(path, list(row.keys()), gzip_flag)
                in_chunk = 0

            writer.writerow(row)
            collected += 1; in_chunk += 1

            if collected % 1000 == 0:
                print(f"[{lang}] {collected} rows…", file=sys.stderr)
                save_state(state_path, {"cursor": cursor, "collected": collected, "chunk_idx": chunk_idx})

            if collected >= target_n:
                break

            if in_chunk >= chunk_size:
                save_state(state_path, {"cursor": cursor, "collected": collected, "chunk_idx": chunk_idx})
                fcsv.close(); writer = None
                in_chunk = 0; chunk_idx += 1

        save_state(state_path, {"cursor": cursor, "collected": collected, "chunk_idx": chunk_idx})

    if fcsv: fcsv.close()
    print(f"[{lang}] DONE: total={collected}, last_part={chunk_idx:03d}", file=sys.stderr)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Lean OpenAlex dump (only fields needed for semantic search + citation graph).")
    ap.add_argument("--email", required=True, help="Ваш e-mail для mailto= (polite pool).")
    ap.add_argument("--outdir", default="./openalex_csv_lean", help="Каталог вывода")
    ap.add_argument("--en", type=int, default=1_000_000, help="Сколько EN статей")
    ap.add_argument("--ru", type=int, default=550_000, help="Сколько RU статей")
    ap.add_argument("--chunk-size", type=int, default=500_00, help="Строк на файл")
    ap.add_argument("--gzip", action="store_true", help="Писать .csv.gz")
    ap.add_argument("--resume", action="store_true", help="Продолжать с сохранённого курсора")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stream_lang("en", args.en, args.email, args.outdir, args.chunk_size, args.gzip, args.resume)
    stream_lang("ru", args.ru, args.email, args.outdir, args.chunk_size, args.gzip, args.resume)

if __name__ == "__main__":
    main()
