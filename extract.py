from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import instructor
import pandas as pd
from batch_pmid import extract_results_passages, fetch_bioc_json, save_json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class Mutation(BaseModel):
    backbone: str | None = Field(
        description="Virus strain/backbone experimentally tested")
    recombinants: list[str] | None = Field(
        description="Recombinant virus strain(s)/backbone(s) experimentally tested, if applicable")
    subtype: str | None = Field(
        description="Virus subtype, such as H1N1 or H5N1")
    proteins: list[str] | None = Field(
        description="Influenza genome segments, such as HA, NA, PB1, PB2, PA, NP, M, or NS")
    mutations: list[str] | None = Field(
        description="Amino-acid mutations in Protein:X111Y format, such as PB2:E627K or HA:N122D. Use Protein:X111- for deletions and Protein:-111Y for insertions. Use a list for multiple mutations tested together in the same experiment.")
    phenotype: str | None= Field(
        description="Relevant mammalian host-adaptation phenotype observed in the same experiment")
    cell_type: str | None = Field(
        description="Cell type used in the experiment to test the mutation, if applicable, such as MDCK, A549, or 293T")
    animal_model: str | None = Field(
        description="Animal model used in the experiment to test the mutation, if applicable. Use specific lineage if known, such as BALB/c")
    receptor_substrates: list[str] | None= Field(
        description="Receptor, glycan, or sialylglycopolymer substrates used in the experiment, such as SAα2,3Gal, SAα2,6Gal, 3'-SLN, 6'-SLN, or human/avian receptor analogs")
    numbering: str | None = Field(
        description="Amino-acid numbering scheme used in the source text, such as H3 or H1 numbering. Only use for HA and NA proteins. If not specified, use null.")
    temperature: str | None = Field(
        description="Temperature of the experiment in the experimental system context, only if explicitly stated in the text. If not specified, use null.")

class ExtractionSchema(BaseModel):
    mutation_records: list[Mutation] = Field(
        description=(
            "Every distinct experimentally tested mutation-phenotype relationship related to host adaptation in the source text"
        )
    )


PROMPT_TEMPLATE = """You are extracting experimentally tested influenza mutation-phenotype relationships from a research paper.

MUTATION EXTRACTION RULES:
* Extract all experimentally tested amino-acid mutation–phenotype relationships in the Results text that are relevant to mammalian host adaptation.
* Exclude background claims, findings from previous studies, predicted or inferred effects, mutations with no observed effect, and general virulence-only outcomes such as pathogenicity, mortality, morbidity, or weight loss.
* Do not infer amino-acid changes from nucleotide changes or guess missing residues or positions.
* Include every relevant mutation or mutation combination, not only the first or most detailed example.
* Create one record for each unique mutation or mutation combination, phenotype, and experimental system.
* If one experiment reports multiple relevant phenotypes for the same mutant, create one record per phenotype.
* If mutations are tested together and the combined phenotype is reported, keep all mutations in one record. Split them only when individual effects are separately tested and reported.
* Create separate records for different tested backbones, strains, subtypes, assays, cell types, animal models, receptor-binding systems, or temperatures.
* Each record must represent only one system type: in vitro or ex vivo cells, in vivo animals, or receptor-binding substrates.
* Extract: backbone, recombinants, subtype, proteins, mutations, phenotype, cell_type, animal_model, receptor_substrates, numbering, and temperature.
* Put the full tested strain or viral backbone in backbone.
* Put each recombinant strain and its tested protein in recombinants, such as A/WSN/1933(HA). Use a list when multiple recombinants are tested together.
* Do not use abbreviated strain names for either backbone or recombinants. Use the full name stated in the text or the most complete available form. Do not use recombinant nicknames, use the official strain names only, unless not stated anywhere in the text
* Put the influenza subtype in subtype, such as H1N1, H3N2, H5N1, or H7N9.
* For reassortants, record the resulting HA/NA subtype only when both HA and NA origins are explicit.
* Put the tested influenza proteins or segments in proteins, using a list when multiple proteins are involved.
* Put mutations in a list using Protein format, such as ["PB2"] or ["HA", "NA"].
* Do not use three-letter notation such as Asp111Glu.
* Format deletions as Protein:X111- and insertions as Protein:-111Y.
* Put only the core experimentally observed host-adaptation effect in phenotype. 
* Use cell_type only for in vitro or ex vivo systems, animal_model only for in vivo systems, and receptor_substrates only for receptors, glycans, or receptor analogs used in binding assays.
* Include the specific cell type, animal strain or lineage, or receptor substrate when stated.
* Use numbering only for explicitly stated HA or NA numbering schemes; otherwise use null.
* Record temperature only when explicitly stated and treat each tested temperature as a separate experiment.
* Use null for every unsupported scalar or list field.
* Do not create duplicate records.
- Extract ONLY what is supported by the text below.


SOURCE USAGE RULES:
- Use the RESULTS TEXT section as the only source for mutation-phenotype claims.
- Use the METHODS TEXT section to fill experimental context fields that were not explicitly stated in the RESULTS TEXT, such as strain/backbone, recombinant, subtype, cell_type, animal_model, receptor_substrates, experimental system, temperature, or HA/NA numbering.
- Do not create a mutation record from METHODS TEXT or alone.
- Do not add a phenotype from METHODS TEXT.
- If METHODS TEXT conflicts with RESULTS TEXT, keep the mutation and phenotype claim grounded in RESULTS TEXT and use METHODS TEXT only for clearly matching experimental context.

--- SOURCE TEXT ---
{content}
"""


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_from_config(config: dict[str, Any], cli_model: str | None) -> str:
    return str(cli_model or config.get("model") or "gpt-4o-mini")


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


def source_text_from_passage_payload(payload: dict[str, Any]) -> str:
    results_text = combined_section_text(
        payload, "combined_results_text", "passages"
    )
    methods_text = combined_section_text(
        payload, "combined_methods_text", "methods_passages"
    )

    sections = [f"--- RESULTS TEXT ---\n{results_text}"]
    if methods_text:
        sections.append(f"--- METHODS TEXT ---\n{methods_text}")

    return "\n\n".join(sections).strip()


def source_text_from_passage_file(path: Path) -> str:
    return source_text_from_passage_payload(load_passage_payload(path))


def optional_float_config(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    return float(value)


def is_content_block_error(error: Exception) -> bool:
    text = str(error).lower()
    block_indicators = [
        "content_filter",
        "content filter",
        "contentfilter",
        "content policy",
        "content management policy",
        "responsibleaipolicyviolation",
        "policy violation",
        "safety system",
        "safety reasons",
        "limited access to this content",
        "invalid prompt",
        "invalid_prompt",
    ]
    if any(indicator in text for indicator in block_indicators):
        return True
    return "content" in text and (
        "blocked" in text
        or "filtered" in text
        or "limited access" in text
        or "safety" in text
    )


def extract_from_text(
    client: instructor.Instructor,
    model: str,
    content: str,
    max_retries: int,
    reasoning_effort: str | None,
    temperature: float | None,
) -> ExtractionSchema:
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        kwargs["temperature"] = temperature

    return client.responses.create(
        model=model,
        max_retries=max_retries,
        response_model=ExtractionSchema,
        input=[
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
        "backbone",
        "recombinants",
        "subtype",
        "proteins",
        "mutations",
        "phenotype",
        "cell_type",
        "animal_model",
        "receptor_substrates",
        "numbering",
        "temperature",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def default_passage_path(pmid: str) -> Path:
    return Path("data") / "passages" / f"{pmid}.json"


def ensure_default_passage_file(pmid: str) -> Path:
    passage_file = default_passage_path(pmid)
    if passage_file.exists():
        return passage_file

    raw_dir = passage_file.parent.parent / "raw_bioc"
    print(f"Fetching PMC BioC passages for PMID {pmid}")
    payload, error = fetch_bioc_json(pmid, raw_dir, timeout=30.0)
    if error:
        if error == "not_found":
            raise RuntimeError(
                f"PMC BioC did not return full-text passages for PMID {pmid}. "
                "This PMID cannot be processed automatically."
            )
        if error == "rate_limited":
            raise RuntimeError(
                "PMC BioC rate limited the request. Wait a minute and rerun "
                "the same command."
            )
        raise RuntimeError(f"Could not fetch PMC BioC JSON for PMID {pmid}: {error}")

    extracted = extract_results_passages(pmid, payload)
    save_json(passage_file, extracted)
    print(f"Saved passage file: {passage_file}")
    return passage_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract influenza mutation-phenotype records from Results plus Methods context."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("openai_config.json"),
        help=(
            "Path to model settings JSON. Defaults to openai_config.json. "
            "Optional keys: model, temperature."
        ),
    )
    parser.add_argument(
        "--fallback-config",
        type=Path,
        default=Path("openai_config_fallback.json"),
        help=(
            "Fallback model config used only when the primary request is "
            "blocked by a content filter. Defaults to openai_config_fallback.json; "
            "the fallback model defaults to o3-mini."
        ),
    )
    parser.add_argument(
        "--pmid",
        default="23750226",
        help="PMID to extract from data/passages/{pmid}.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Overrides config; defaults to gpt-4o-mini.",
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
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
        help=(
            "Model reasoning effort. Use none to omit the parameter."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Instructor retry count for schema validation failures.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config_path = args.config if args.config.exists() else None
    config = load_config(config_path)
    model = model_from_config(config, args.model)
    temperature = optional_float_config(config, "temperature")

    passage_file = ensure_default_passage_file(args.pmid)
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
    content = source_text_from_passage_payload(passage_payload)
    if not content:
        raise RuntimeError(f"No Results text found in {passage_file}")

    print(f"Results characters available: {results_chars}")
    print(f"Methods characters available: {methods_chars}")
    print(f"Characters sent to model: {len(content)}")

    client = instructor.from_openai(
        OpenAI(
            base_url=os.environ["PROXY_BASE_URL"].rstrip("/"),
            api_key="unused",  # Authentication is via x-functions-key.
            default_headers={"x-functions-key": os.environ["FUNCTION_HOST_KEY"]},
        ),
        mode=instructor.Mode.RESPONSES_TOOLS,
    )
    reasoning_effort = (
        None if args.reasoning_effort == "none" else args.reasoning_effort
    )
    final_model = model
    used_fallback = False

    rows: list[dict[str, Any]] = []
    try:
        extraction = extract_from_text(
            client=client,
            model=model,
            content=content,
            max_retries=args.max_retries,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        rows = extraction_rows(source_pmid, source_pmcid, extraction)
        save_outputs(rows, output_csv, output_json)
    except Exception as exc:
        content_blocked = is_content_block_error(exc)
        fallback_config = {}
        fallback_model = None
        if content_blocked:
            if args.fallback_config.exists():
                fallback_config = load_config(args.fallback_config)
            fallback_model = str(fallback_config.get("model") or "o3-mini")

        if content_blocked and fallback_model != model:
            print(
                "WARNING: Primary model was blocked by content filtering: "
                f"{model}"
            )
            print(
                "WARNING: Retrying once with fallback model: "
                f"{fallback_model}"
            )

            fallback_temperature = optional_float_config(
                fallback_config, "temperature"
            )
            try:
                extraction = extract_from_text(
                    client=client,
                    model=fallback_model,
                    content=content,
                    max_retries=args.max_retries,
                    reasoning_effort=None,
                    temperature=fallback_temperature,
                )
                rows = extraction_rows(source_pmid, source_pmcid, extraction)
                save_outputs(rows, output_csv, output_json)
            except Exception:
                raise

            final_model = fallback_model
            used_fallback = True
        elif content_blocked:
            print(
                "WARNING: Primary model was blocked by content filtering, "
                "but no distinct fallback model was available: "
                f"{args.fallback_config}"
            )
            raise
        else:
            raise

    print(f"Extracted {len(rows)} mutation records for PMID {source_pmid}")
    if used_fallback:
        print(
            "WARNING: Final output was generated with fallback model: "
            f"{final_model}"
        )
    else:
        print(
            "Final output was generated with primary model: "
            f"{final_model}"
        )
    print(f"CSV: {output_csv}")
    print(f"JSON: {output_json}")


if __name__ == "__main__":
    main()
