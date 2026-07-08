from __future__ import annotations

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BIOC_API_TEMPLATE = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
    "BioC_json/{pmid}/unicode"
)
RESULTS_SECTION_TYPE = "RESULTS"


def normalize_pmid_token(token: Any) -> str | None:
    """Return a canonical PMID string, including Excel scientific notation."""
    if token is None:
        return None

    text = str(token).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    if re.fullmatch(r"\d+", text):
        return text

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None

    if value < 0 or value != value.to_integral_value():
        return None

    return format(value.quantize(Decimal("1")), "f")


def split_pmid_cell(cell_value: Any) -> list[str]:
    if cell_value is None:
        return []

    raw_tokens = re.split(r"[;,]", str(cell_value))
    pmids: list[str] = []
    for token in raw_tokens:
        pmid = normalize_pmid_token(token)
        if pmid:
            pmids.append(pmid)
    return pmids


def ordered_unique_pmids(cell_values: list[Any]) -> list[str]:
    seen: set[str] = set()
    pmids: list[str] = []
    for cell_value in cell_values:
        for pmid in split_pmid_cell(cell_value):
            if pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)
    return pmids


def load_pmids(sheet_path: Path, pmid_column: str = "PMID") -> list[str]:
    sheet = pd.read_excel(sheet_path)
    if pmid_column not in sheet.columns:
        available = ", ".join(str(column) for column in sheet.columns)
        raise ValueError(
            f"Expected a {pmid_column!r} column in {sheet_path}. "
            f"Available columns: {available}"
        )

    return ordered_unique_pmids(sheet[pmid_column].tolist())


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_bioc_json(
    pmid: str,
    raw_dir: Path,
    timeout: float,
    refresh: bool = False,
) -> tuple[Any | None, str | None]:
    raw_path = raw_dir / f"{pmid}.json"
    if raw_path.exists() and not refresh:
        return load_json(raw_path), None

    url = BIOC_API_TEMPLATE.format(pmid=pmid)
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            return None, "not_found"
        response.raise_for_status()
        if "No result can be found" in response.text:
            return None, "not_found"
        payload = response.json()
    except requests.RequestException as exc:
        return None, f"http_error: {exc}"
    except ValueError as exc:
        return None, f"invalid_json: {exc}"

    save_json(raw_path, payload)
    return payload, None


def iter_bioc_documents(payload: Any) -> list[dict[str, Any]]:
    collections = payload if isinstance(payload, list) else [payload]
    documents: list[dict[str, Any]] = []
    for collection in collections:
        if not isinstance(collection, dict):
            continue
        for document in collection.get("documents", []):
            if isinstance(document, dict):
                documents.append(document)
    return documents


def extract_results_passages(pmid: str, payload: Any) -> dict[str, Any]:
    documents = iter_bioc_documents(payload)
    extracted_documents: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []

    for document in documents:
        pmc_id = document.get("id")
        document_passages: list[dict[str, Any]] = []
        for passage in document.get("passages", []):
            if not isinstance(passage, dict):
                continue

            infons = passage.get("infons") or {}
            section_type = str(infons.get("section_type", "")).upper()
            if section_type != RESULTS_SECTION_TYPE:
                continue

            passage_record = {
                "pmid": pmid,
                "pmc_id": pmc_id,
                "section_type": section_type,
                "passage_type": infons.get("type"),
                "section": infons.get("section"),
                "offset": passage.get("offset"),
                "text": passage.get("text", ""),
            }
            passages.append(passage_record)
            document_passages.append(passage_record)

        extracted_documents.append(
            {
                "pmc_id": pmc_id,
                "passage_count": len(document_passages),
                "passages": document_passages,
            }
        )

    return {
        "pmid": pmid,
        "source": "PMC BioC",
        "document_count": len(documents),
        "passage_count": len(passages),
        "documents": extracted_documents,
        "passages": passages,
        "combined_results_text": "\n\n".join(
            passage["text"] for passage in passages if passage.get("text")
        ),
    }


def run_pipeline(
    sheet_path: Path,
    out_dir: Path,
    limit: int | None = None,
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    refresh: bool = False,
) -> dict[str, Any]:
    raw_dir = out_dir / "raw_bioc"
    passages_dir = out_dir / "passages"

    pmids = load_pmids(sheet_path)
    if limit is not None:
        pmids = pmids[:limit]

    save_json(
        out_dir / "pmids.json",
        {
            "source_sheet": str(sheet_path),
            "pmid_count": len(pmids),
            "pmids": pmids,
        },
    )

    report: dict[str, Any] = {
        "source_sheet": str(sheet_path),
        "attempted_count": len(pmids),
        "successes": [],
        "unavailable_pmids": [],
        "no_results_section": [],
        "errors": [],
    }

    for index, pmid in enumerate(pmids, start=1):
        print(f"[{index}/{len(pmids)}] Fetching PMID {pmid}")
        payload, error = fetch_bioc_json(pmid, raw_dir, timeout, refresh=refresh)
        if error:
            bucket = "unavailable_pmids" if error == "not_found" else "errors"
            report[bucket].append({"pmid": pmid, "error": error})
            continue

        documents = iter_bioc_documents(payload)
        if not documents:
            report["unavailable_pmids"].append(
                {"pmid": pmid, "error": "no_documents"}
            )
            continue

        extracted = extract_results_passages(pmid, payload)
        save_json(passages_dir / f"{pmid}.json", extracted)

        success_record = {
            "pmid": pmid,
            "pmc_ids": [document.get("id") for document in documents],
            "passage_count": extracted["passage_count"],
        }
        report["successes"].append(success_record)

        if extracted["passage_count"] == 0:
            report["no_results_section"].append(pmid)

        if sleep_seconds > 0 and index < len(pmids):
            time.sleep(sleep_seconds)

    save_json(out_dir / "run_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch PMC BioC articles by PMID and save Results-only passages."
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=Path("Curated_Mutation_Dataset.xlsx"),
        help="Path to the curated workbook containing a PMID column.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory where JSON artifacts will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds per PMID.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download raw BioC JSON even when a cached file exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(
        sheet_path=args.sheet,
        out_dir=args.out_dir,
        limit=args.limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        refresh=args.refresh,
    )
    print(
        "Done. "
        f"successes={len(report['successes'])}, "
        f"unavailable={len(report['unavailable_pmids'])}, "
        f"errors={len(report['errors'])}, "
        f"no_results={len(report['no_results_section'])}"
    )


if __name__ == "__main__":
    main()
