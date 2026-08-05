# FLAME: FLu Atlas of Mutation Effects
## Mutation Extraction Pipeline

This repository contains a literature-mining pipeline for extracting experimentally tested influenza A mutation-phenotype-experimental context relationships relevant to mammalian host adaptation. The pipeline retrieves PMC BioC article text, isolates Results and Methods passages, sends the source text along with a prompt to an Azure OpenAI GPT deployment, validates structured outputs with Instructor and Pydantic, and writes mutation records to CSV and JSON.

## Overview

The pipeline is designed to extract mutation records from full-text biomedical articles using a constrained prompt and a typed output schema. Mutation-phenotype claims are extracted only from Results text. Methods text is included only to fill experimental context fields, such as viral backbone, recombinant strain, cell type, animal model, receptor substrate, numbering scheme, or temperature.

Data flow:

```text
PMID(s) -> PMC BioC JSON -> Results and Methods passages -> GPT extraction with Instructor and Pydantic validation -> validated ExtractionSchema object -> CSV and JSON mutation records
```

## Repository Contents

- `batch_pmid.py`: Downloads PMC BioC JSON by PMID and saves Results/Methods passage files.
- `extract.py`: Runs the GPT extraction and writes type validated CSV/JSON outputs.
- `requirements.txt`: Python dependencies.
- `tests/`: Unit tests for PMID parsing, passage filtering, and extraction helper behavior.
- `data/`: Generated BioC, passages, and extractions.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

`extract.py` expects Azure OpenAI credentials and deployment information from `openai_config.json` by default. You can also pass a different config file with `--config`.

Example config shape:

```json
{
  "subscription_key": "YOUR_AZURE_OPENAI_KEY",
  "endpoint_url": "https://YOUR_RESOURCE.openai.azure.com/",
  "api_version": "YOUR_API_VERSION",
  "deployment": "YOUR_DEPLOYMENT_NAME",
  "temperature": 0.0
}
```

Supported credential environment variables include:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

Do not commit real API keys or local credential files.

## Fetch BioC Passages in Batch

Use `batch_pmid.py` to prefetch passages for many PMIDs, usually from a curated spreadsheet:
* Spreadsheet must contain "PMID" column.

```powershell
python batch_pmid.py --sheet "your_pmid_spreadsheet"
```

Useful options:

- `--sheet`: Spreadsheet path.
- `--out-dir`: Output directory. Defaults to `data`.
- `--limit`: Limit the number of PMIDs for testing.
- `--refresh`: Re-download raw BioC JSON even if cached files already exist.

Generated passage outputs are written to:

- `data/raw_bioc/{pmid}.json`
- `data/passages/{pmid}.json`
- `data/pmids.json`
- `data/run_report.json`

## Run Extraction

Run structured extraction for one PMID. If `data/passages/{pmid}.json` does not exist, `extract.py` downloads the PMC BioC JSON, saves the Results/Methods passage file, and then runs extraction.

```powershell
python extract.py --pmid 23750226
```

If PMC BioC rate limits the request, wait briefly and rerun the same command. If PMC BioC does not have full-text passages for the PMID, that PMID cannot be processed automatically.

Useful options:

- `--config`: Azure OpenAI config path. Defaults to `openai_config.json`.
- `--fallback-config`: Optional fallback config (preferably to a smaller model) used after content-filter blocks. Defaults to `openai_config_fallback.json`.
- `--deployment`: Override the deployment name from the config file.
- `--output-csv`: Custom CSV output path.
- `--output-json`: Custom JSON output path.
- `--reasoning-effort`: GPT reasoning effort: `none`, `low`, `medium`, or `high`. Defaults to `low`.
- `--max-retries`: Instructor retry count for schema validation failures. Defaults to `2`.

## Output Fields

Extraction outputs are saved by default to:

- `data/extractions/{pmid}_mutations.csv`
- `data/extractions/{pmid}_mutations.json`

Each extracted record contains:

- `pmid`: PubMed ID.
- `pmcid`: PMC article ID, when available.
- `backbone`: Virus strain or experimental backbone.
- `recombinants`: Recombinant virus strain(s), when applicable.
- `subtype`: Influenza subtype, such as H1N1 or H5N1.
- `proteins`: Influenza proteins or genome segments tested.
- `mutations`: Tested amino-acid mutations.
- `phenotype`: Experimentally observed mammalian host-adaptation phenotype.
- `cell_type`: In vitro or ex vivo cell type.
- `animal_model`: In vivo animal model.
- `receptor_substrates`: Receptor, glycan, or receptor analog substrates.
- `numbering`: HA or NA numbering scheme, when stated.
- `temperature`: Experimental temperature, when stated.

## Structured Extraction

The extraction schema is defined in `extract.py` using Pydantic models:

- `Mutation`: One experimentally tested mutation-phenotype relationship.
- `ExtractionSchema`: A list of `Mutation` records.

Instructor wraps the Azure OpenAI chat-completion call, requests an `ExtractionSchema` response, validates the model response with Pydantic, and retries invalid responses when possible. The final validated object is flattened into tabular rows before export.

## Tests

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

Current tests cover PMID parsing, BioC passage filtering, content-filter error detection, and optional config parsing.

## Limitations

- Articles must be available through PMC BioC for automated full-text retrieval.
- Passage selection depends on BioC section metadata quality.
- BioC does not retrieve figures (Figure captions are included however).
- Model outputs should be manually reviewed before inclusion in a curated scientific dataset.

## Citation Notes

Relevant tools and resources used by this pipeline include PMC BioC, Azure OpenAI, Instructor, Pydantic, pandas, and requests.
