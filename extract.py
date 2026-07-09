from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import instructor
from instructor.core.client import HookName
import pandas as pd
from openai import AzureOpenAI
from pydantic import BaseModel, Field


class Mutation(BaseModel):
    strain: str | None = Field(
        description="Virus strain/backbone experimentally tested")
    subtype: str | None = Field(
        description="Virus subtype, such as H1N1 or H5N1")
    protein: str | None = Field(
        description="Influenza genome segment, such as HA, NA, PB1, PB2, PA, NP, M, or NS")
    mutation: list[str] = Field(
        description="Amino-acid mutation in X111Y format")
    phenotypes: list[str] = Field(
        description="Relevant mammalian host-adaptation phenotypes observed in the same experiment")
    cell_type: str | None = Field(
        description="Cell type used in the experiment to test the mutation, if applicable, such as MDCK, A549, or 293T")
    animal_model: str | None = Field(
        description="Animal model used in the experiment to test the mutation, if applicable. Use specific lineage if known, such as BALB/c")
    


class ExtractionSchema(BaseModel):
    mutation_records: list[Mutation] = Field(
        description=(
            "Every distinct experimentally tested mutation-phenotype relationship related to"
            "host adaptationin the source text"
        )
    )


class UsageTotals(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    response_count: int = 0


PROMPT_TEMPLATE = """You are extracting experimentally tested influenza mutation-phenotype relationships from a research paper.

MUTATION EXTRACTION RULES:
- Only extract mutations that relate to mammalian host adaptation.
- Do not extract general virulence-only outcomes such as increased virulence, weight loss, mortality, morbidity, or pathogenicity.
- Do not report mutations that had no observed effect.
- If the same experiment reports multiple relevant host-adaptation phenotypes for the same mutation or mutation combination, create one record and put each phenotype in the phenotypes list.
- If an experiment tests multiple mutations together and reports the phenotype of the combined mutant, create one record with all tested amino-acid mutations in the mutation list. Do not split combined-mutant phenotypes into separate single-mutation records unless the text separately reports effects for the individual mutations.
- Create a different record for each distinct experiment, including different cell type, animal model, viral backbone, strain, subtype, assay, or experimental system.
- Extract only experimentally tested relationships from this paper.
- Do not extract background claims, literature review claims, or mutations cited from previous studies.
- For each record, extract tested strain/backbone, subtype, protein, amino-acid mutation list, phenotype list, cell type, and animal model.
- Do not extract PMID or PMCID from the source text; those identifiers are added separately from pipeline metadata.
- Use the mutation field as a list of amino-acid mutations in X111Y format, such as ["E627K"] or ["G590S", "Q591R"].
- Use the phenotypes field as a list of experimentally observed mammalian host-adaptation phenotypes from the same experiment.
- Use cell_type for in-vitro or ex-vivo cell systems, such as MDCK, A549, 293T, Calu-3, or primary cells.
- Use animal_model for in-vivo animal systems, including the specific lineage or strain when stated, such as BALB/c mice, ferrets, guinea pigs, or swine.
- Put strain/backbone in strain, such as A/Zhejiang/DTID-ZJU01/2013 or A/turkey/Egypt/AR1507/2016.
- Put subtype in subtype, such as H1N1, H3N2, H5N1, or H7N9.
- Put the influenza protein or segment in protein, such as PB2, PB1, PA, HA, NA, NP, M, NS.
- Extract ONLY what is supported by the text below.
- If a nullable field is not present, use null rather than guessing.
- If a list field has no supported values, use an empty list rather than guessing.
- Do not infer an amino-acid mutation from a nucleotide change unless the amino-acid consequence is explicitly stated.
- There should be absolutely no duplicate entries.

SOURCE USAGE RULES:
- Use the RESULTS TEXT section as the only source for mutation-phenotype claims.
- Use the METHODS TEXT section only to fill experimental context fields such as strain/backbone, subtype, cell_type, animal_model, assay, or experimental system.
- Do not create a mutation record from METHODS TEXT alone.
- Do not add a phenotype from METHODS TEXT unless the same phenotype claim is supported in RESULTS TEXT.
- If METHODS TEXT conflicts with RESULTS TEXT, keep the mutation and phenotype claim grounded in RESULTS TEXT and use METHODS TEXT only for clearly matching experimental context.

--- SOURCE TEXT ---
{content}
"""


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_value(
    config: dict[str, Any],
    keys: list[str],
    env_names: list[str],
    label: str,
) -> str:
    for key in keys:
        value = config.get(key)
        if value:
            return str(value)

    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value

    raise RuntimeError(
        f"Missing {label}. Add one of {keys} to your config JSON "
        f"or set one of {env_names}."
    )


def create_client(config: dict[str, Any]) -> instructor.Instructor:
    azure_client = AzureOpenAI(
        api_key=config_value(
            config,
            ["api_key", "subscription_key", "azure_openai_api_key"],
            ["AZURE_OPENAI_API_KEY"],
            "Azure OpenAI API key",
        ),
        azure_endpoint=config_value(
            config,
            ["endpoint", "endpoint_url", "azure_endpoint", "azure_openai_endpoint"],
            ["AZURE_OPENAI_ENDPOINT"],
            "Azure OpenAI endpoint",
        ),
        api_version=config_value(
            config,
            ["api_version", "azure_openai_api_version"],
            ["AZURE_OPENAI_API_VERSION"],
            "Azure OpenAI API version",
        ),
    )
    return instructor.from_openai(azure_client)


def deployment_from_config(config: dict[str, Any], cli_deployment: str | None) -> str:
    if cli_deployment:
        return cli_deployment
    return config_value(
        config,
        ["deployment", "deployment_name", "azure_openai_deployment"],
        ["AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME"],
        "Azure OpenAI deployment",
    )


def load_passage_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pmcid_from_passage_payload(payload: dict[str, Any]) -> str | None:
    for document in payload.get("documents", []):
        pmcid = document.get("pmc_id")
        if pmcid:
            return str(pmcid)

    for passage in payload.get("passages", []):
        pmcid = passage.get("pmc_id")
        if pmcid:
            return str(pmcid)

    return None


def combined_section_text(
    payload: dict[str, Any],
    combined_key: str,
    passages_key: str,
) -> str:
    text = payload.get(combined_key)
    if text:
        return str(text).strip()

    return "\n\n".join(
        passage.get("text", "")
        for passage in payload.get(passages_key, [])
        if passage.get("text")
    ).strip()


def source_text_from_passage_payload(
    payload: dict[str, Any],
    max_chars: int | None = None,
    max_methods_chars: int | None = 15000,
) -> str:
    results_text = combined_section_text(
        payload, "combined_results_text", "passages"
    )
    methods_text = combined_section_text(
        payload, "combined_methods_text", "methods_passages"
    )
    if max_methods_chars is not None and len(methods_text) > max_methods_chars:
        methods_text = methods_text[:max_methods_chars]

    sections = [f"--- RESULTS TEXT ---\n{results_text}"]
    if methods_text:
        sections.append(f"--- METHODS TEXT ---\n{methods_text}")

    text = "\n\n".join(sections).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def source_text_from_passage_file(path: Path, max_chars: int | None = None) -> str:
    return source_text_from_passage_payload(
        load_passage_payload(path), max_chars=max_chars
    )


def value_from(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def collect_usage_from_response(response: Any, totals: UsageTotals) -> None:
    usage = value_from(response, "usage")
    if usage is None:
        return

    prompt_tokens = int(value_from(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(value_from(usage, "completion_tokens", 0) or 0)
    total_tokens = int(value_from(usage, "total_tokens", 0) or 0)

    completion_details = value_from(usage, "completion_tokens_details")
    prompt_details = value_from(usage, "prompt_tokens_details")
    reasoning_tokens = int(value_from(completion_details, "reasoning_tokens", 0) or 0)
    cached_prompt_tokens = int(value_from(prompt_details, "cached_tokens", 0) or 0)

    totals.prompt_tokens += prompt_tokens
    totals.completion_tokens += completion_tokens
    totals.total_tokens += total_tokens
    totals.reasoning_tokens += reasoning_tokens
    totals.cached_prompt_tokens += cached_prompt_tokens
    totals.response_count += 1


def attach_usage_tracker(client: instructor.Instructor) -> UsageTotals:
    totals = UsageTotals()
    client.on(
        HookName.COMPLETION_RESPONSE,
        lambda response: collect_usage_from_response(response, totals),
    )
    return totals


def token_cost(config: dict[str, Any], key: str) -> float:
    value = config.get(key, 0)
    if value in (None, ""):
        return 0.0
    return float(value)


def read_running_totals(spend_csv: Path) -> tuple[int, int, int, float]:
    if not spend_csv.exists():
        return 0, 0, 0, 0.0

    with spend_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return 0, 0, 0, 0.0

    last = rows[-1]
    return (
        int(float(last.get("running_prompt_tokens") or 0)),
        int(float(last.get("running_completion_tokens") or 0)),
        int(float(last.get("running_total_tokens") or 0)),
        float(last.get("running_cost_usd") or 0.0),
    )


def spend_csv_fieldnames(spend_csv: Path, row: dict[str, Any]) -> list[str]:
    new_fields = list(row.keys())
    if not spend_csv.exists():
        return new_fields

    with spend_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    missing_fields = [field for field in new_fields if field not in existing_fields]
    if not missing_fields:
        return existing_fields

    merged_fields = existing_fields + missing_fields
    with spend_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(existing_rows)
    return merged_fields


def append_spend_row(
    spend_csv: Path,
    *,
    pmid: str,
    passage_file: Path,
    deployment: str,
    status: str,
    record_count: int,
    usage: UsageTotals,
    config: dict[str, Any],
    max_tokens: int,
    reasoning_effort: str,
    max_chars: int | None,
    max_methods_chars: int | None,
    max_retries: int,
    error: Exception | None = None,
) -> dict[str, Any]:
    prompt_cost_per_token = token_cost(config, "prompt_token_cost")
    completion_cost_per_token = token_cost(config, "completion_token_cost")
    prompt_cost = usage.prompt_tokens * prompt_cost_per_token
    completion_cost = usage.completion_tokens * completion_cost_per_token
    total_cost = prompt_cost + completion_cost

    (
        running_prompt_tokens,
        running_completion_tokens,
        running_total_tokens,
        running_cost,
    ) = read_running_totals(spend_csv)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pmid": pmid,
        "passage_file": str(passage_file),
        "deployment": deployment,
        "status": status,
        "record_count": record_count,
        "response_count": usage.response_count,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cached_prompt_tokens": usage.cached_prompt_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_cost_usd": f"{prompt_cost:.8f}",
        "completion_cost_usd": f"{completion_cost:.8f}",
        "total_cost_usd": f"{total_cost:.8f}",
        "running_prompt_tokens": running_prompt_tokens + usage.prompt_tokens,
        "running_completion_tokens": running_completion_tokens
        + usage.completion_tokens,
        "running_total_tokens": running_total_tokens + usage.total_tokens,
        "running_cost_usd": f"{running_cost + total_cost:.8f}",
        "prompt_token_cost": prompt_cost_per_token,
        "completion_token_cost": completion_cost_per_token,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "max_chars": max_chars,
        "max_methods_chars": max_methods_chars,
        "max_retries": max_retries,
        "error_type": type(error).__name__ if error else "",
        "error_message": str(error).replace("\n", " ")[:1000] if error else "",
    }

    spend_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not spend_csv.exists()
    fieldnames = spend_csv_fieldnames(spend_csv, row)
    with spend_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def extract_from_text(
    client: instructor.Instructor,
    deployment: str,
    content: str,
    max_tokens: int,
    max_retries: int,
    reasoning_effort: str | None,
) -> ExtractionSchema:
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    return client.chat.completions.create(
        model=deployment,
        max_completion_tokens=max_tokens,
        max_retries=max_retries,
        response_model=ExtractionSchema,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(content=content),
            }
        ],
        **kwargs,
    )


def extraction_rows(
    pmid: str, pmcid: str | None, extraction: ExtractionSchema
) -> list[dict[str, Any]]:
    rows = []
    for record in extraction.mutation_records:
        row = {"pmid": pmid, "pmcid": pmcid}
        row.update(record.model_dump())
        rows.append(row)
    return rows


def save_outputs(rows: list[dict[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "pmid",
        "pmcid",
        "strain",
        "subtype",
        "protein",
        "mutation",
        "phenotypes",
        "cell_type",
        "animal_model"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def default_passage_path(pmid: str) -> Path:
    return Path("data") / "passages" / f"{pmid}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract influenza mutation-phenotype records from Results plus Methods context."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("openai_config.json"),
        help=(
            "Path to Azure OpenAI config JSON. Defaults to openai_config.json. "
            "Expected keys: subscription_key, endpoint_url, api_version, deployment."
        ),
    )
    parser.add_argument(
        "--pmid",
        default="23750226",
        help="PMID to extract from data/passages/{pmid}.json.",
    )
    parser.add_argument(
        "--passage-file",
        type=Path,
        default=None,
        help="Optional explicit passage JSON file. Overrides --pmid path lookup.",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Azure OpenAI deployment name. Overrides the config JSON value.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to data/extractions/{pmid}_mutations.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON output path. Defaults to data/extractions/{pmid}_mutations.json.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help=(
            "Optional maximum labeled Results plus Methods source characters to "
            "send. Defaults to no truncation."
        ),
    )
    parser.add_argument(
        "--max-methods-chars",
        type=int,
        default=None,
        help="Optional maximum Methods characters to include. Defaults to no truncation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help=(
            "Maximum completion tokens, including hidden reasoning tokens, "
            "for the extraction response."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="low",
        help=(
            "Azure GPT-5 reasoning effort. Use low to reduce hidden reasoning "
            "tokens; use none to omit the parameter."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Instructor retry count for schema validation failures.",
    )
    parser.add_argument(
        "--spend-csv",
        type=Path,
        default=Path("data") / "token_spend.csv",
        help="Append token and cost tracking rows to this CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.exists() else None
    config = load_config(config_path)
    deployment = deployment_from_config(config, args.deployment)

    passage_file = args.passage_file or default_passage_path(args.pmid)
    output_csv = args.output_csv or (
        Path("data") / "extractions" / f"{args.pmid}_mutations.csv"
    )
    output_json = args.output_json or (
        Path("data") / "extractions" / f"{args.pmid}_mutations.json"
    )

    passage_payload = load_passage_payload(passage_file)
    source_pmid = str(passage_payload.get("pmid") or args.pmid)
    source_pmcid = pmcid_from_passage_payload(passage_payload)
    results_chars = len(
        combined_section_text(passage_payload, "combined_results_text", "passages")
    )
    methods_chars = len(
        combined_section_text(
            passage_payload, "combined_methods_text", "methods_passages"
        )
    )
    source_without_total_cap = source_text_from_passage_payload(
        passage_payload,
        max_chars=None,
        max_methods_chars=args.max_methods_chars,
    )
    content = source_text_from_passage_payload(
        passage_payload,
        max_chars=args.max_chars,
        max_methods_chars=args.max_methods_chars,
    )
    if not content:
        raise RuntimeError(f"No Results text found in {passage_file}")

    print(f"Results characters available: {results_chars}")
    print(f"Methods characters available: {methods_chars}")
    if args.max_methods_chars is not None and methods_chars > args.max_methods_chars:
        print(
            "WARNING: Methods text was truncated by "
            f"--max-methods-chars={args.max_methods_chars}."
        )
    if args.max_chars is not None and len(source_without_total_cap) > args.max_chars:
        print(
            "WARNING: Combined source text was truncated by "
            f"--max-chars={args.max_chars}."
        )
    print(f"Characters sent to model: {len(content)}")

    client = create_client(config)
    usage = attach_usage_tracker(client)
    reasoning_effort = (
        None if args.reasoning_effort == "none" else args.reasoning_effort
    )

    rows: list[dict[str, Any]] = []
    try:
        extraction = extract_from_text(
            client=client,
            deployment=deployment,
            content=content,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            reasoning_effort=reasoning_effort,
        )
        rows = extraction_rows(source_pmid, source_pmcid, extraction)
        save_outputs(rows, output_csv, output_json)
    except Exception as exc:
        spend_row = append_spend_row(
            args.spend_csv,
            pmid=source_pmid,
            passage_file=passage_file,
            deployment=deployment,
            status="failed",
            record_count=0,
            usage=usage,
            config=config,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            max_chars=args.max_chars,
            max_methods_chars=args.max_methods_chars,
            max_retries=args.max_retries,
            error=exc,
        )
        print(f"Spend CSV: {args.spend_csv}")
        print(f"This run cost: ${spend_row['total_cost_usd']}")
        print(f"Running total: ${spend_row['running_cost_usd']}")
        raise

    spend_row = append_spend_row(
        args.spend_csv,
        pmid=source_pmid,
        passage_file=passage_file,
        deployment=deployment,
        status="success",
        record_count=len(rows),
        usage=usage,
        config=config,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_chars=args.max_chars,
        max_methods_chars=args.max_methods_chars,
        max_retries=args.max_retries,
    )

    print(f"Extracted {len(rows)} mutation records for PMID {source_pmid}")
    print(f"CSV: {output_csv}")
    print(f"JSON: {output_json}")
    print(f"Spend CSV: {args.spend_csv}")
    print(f"This run cost: ${spend_row['total_cost_usd']}")
    print(f"Running total: ${spend_row['running_cost_usd']}")


if __name__ == "__main__":
    main()
