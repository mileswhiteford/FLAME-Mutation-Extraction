import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import extract
from batch_pmid import (
    extract_results_passages,
    is_methods_source_passage,
    is_results_source_passage,
    normalize_pmid_token,
    ordered_unique_pmids,
    split_pmid_cell,
)
from extract import ensure_default_passage_file, is_content_block_error, optional_float_config


class PmidParsingTests(unittest.TestCase):
    def test_plain_pmid(self):
        self.assertEqual(normalize_pmid_token("23750226"), "23750226")

    def test_scientific_notation_pmid(self):
        self.assertEqual(normalize_pmid_token("2.3750226E7"), "23750226")

    def test_comma_separated_cell(self):
        self.assertEqual(
            split_pmid_cell("23077313, 23750226"),
            ["23077313", "23750226"],
        )

    def test_semicolon_separated_cell(self):
        self.assertEqual(
            split_pmid_cell("23077313; 23750226"),
            ["23077313", "23750226"],
        )

    def test_empty_cells(self):
        self.assertEqual(split_pmid_cell(None), [])
        self.assertEqual(split_pmid_cell(""), [])
        self.assertEqual(split_pmid_cell("nan"), [])

    def test_ordered_unique_pmids(self):
        self.assertEqual(
            ordered_unique_pmids(["2.3750226E7", "23077313, 23750226", None]),
            ["23750226", "23077313"],
        )

class ResultsExtractionTests(unittest.TestCase):
    def test_extracts_only_results_passages(self):
        payload = [
            {
                "documents": [
                    {
                        "id": "PMC123",
                        "passages": [
                            {
                                "offset": 1,
                                "infons": {
                                    "section_type": "METHODS",
                                    "type": "paragraph",
                                    "section": "Methods",
                                },
                                "text": "Methods text",
                            },
                            {
                                "offset": 2,
                                "infons": {
                                    "section_type": "RESULTS",
                                    "type": "paragraph",
                                    "section": "Results",
                                },
                                "text": "Results text",
                            },
                            {
                                "offset": 3,
                                "infons": {
                                    "section_type": "DISCUSS",
                                    "type": "paragraph",
                                    "section": "Discussion",
                                },
                                "text": "Discussion text",
                            },
                            {
                                "offset": 4,
                                "infons": {
                                    "section_type": "FIG",
                                    "type": "fig_caption",
                                },
                                "text": "Figure caption text",
                            },
                            {
                                "offset": 5,
                                "infons": {
                                    "section_type": "TABLE",
                                    "type": "table_caption",
                                },
                                "text": "Table caption text",
                            },
                        ],
                    }
                ]
            }
        ]

        extracted = extract_results_passages("23750226", payload)

        self.assertEqual(extracted["passage_count"], 2)
        self.assertEqual(extracted["methods_passage_count"], 1)
        self.assertEqual(extracted["passages"][0]["text"], "Results text")
        self.assertEqual(extracted["passages"][1]["text"], "Figure caption text")
        self.assertEqual(extracted["methods_passages"][0]["text"], "Methods text")
        self.assertEqual(
            extracted["combined_results_text"],
            "Results text\n\nFigure caption text",
        )
        self.assertEqual(extracted["combined_methods_text"], "Methods text")

    def test_results_source_passages_include_figure_captions(self):
        self.assertTrue(is_results_source_passage("RESULTS", "paragraph"))
        self.assertTrue(is_results_source_passage("FIG", "fig_caption"))
        self.assertTrue(is_results_source_passage("FIG", "fig_title_caption"))
        self.assertTrue(
            is_results_source_passage("DISCUSS", "paragraph", "Results/Discussion")
        )
        self.assertTrue(
            is_results_source_passage("DISCUSS", "paragraph", "Discussion/Results")
        )
        self.assertFalse(is_results_source_passage("TABLE", "table_caption"))
        self.assertFalse(is_results_source_passage("REF", "title_caption"))

    def test_methods_source_passages_include_experimental_procedures(self):
        self.assertTrue(is_methods_source_passage("METHODS"))
        self.assertTrue(
            is_methods_source_passage("DISCUSS", "Experimental Procedures")
        )
        self.assertFalse(is_methods_source_passage("DISCUSS", "Discussion"))


class ExtractFallbackTests(unittest.TestCase):
    def test_detects_content_block_errors(self):
        error = RuntimeError(
            "Error code: 400 - ResponsibleAIPolicyViolation: "
            "request blocked by content filter"
        )

        self.assertTrue(is_content_block_error(error))

    def test_detects_invalid_prompt_safety_block(self):
        error = RuntimeError(
            "Error code: 400 - {'error': {'message': \"Invalid prompt: "
            "we've limited access to this content for safety reasons.\", "
            "'code': 'invalid_prompt'}}"
        )

        self.assertTrue(is_content_block_error(error))

    def test_does_not_treat_generic_bad_request_as_content_block(self):
        error = RuntimeError(
            "Error code: 400 - Unrecognized request argument supplied: "
            "reasoning_effort"
        )

        self.assertFalse(is_content_block_error(error))

    def test_optional_float_config(self):
        self.assertEqual(optional_float_config({"temperature": "0.0"}, "temperature"), 0.0)
        self.assertIsNone(optional_float_config({}, "temperature"))


class ExtractPassageFetchTests(unittest.TestCase):
    def test_uses_existing_default_passage_file(self):
        original_fetch = extract.fetch_bioc_json
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                passage_file = Path("data") / "passages" / "12345.json"
                passage_file.parent.mkdir(parents=True)
                passage_file.write_text('{"pmid": "12345"}', encoding="utf-8")

                def fail_fetch(*args, **kwargs):
                    self.fail("Existing passage file should not be fetched again")

                extract.fetch_bioc_json = fail_fetch
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ensure_default_passage_file("12345"), passage_file)
            finally:
                extract.fetch_bioc_json = original_fetch
                os.chdir(original_cwd)

    def test_fetches_missing_default_passage_file(self):
        original_fetch = extract.fetch_bioc_json
        payload = [
            {
                "documents": [
                    {
                        "id": "PMC123",
                        "passages": [
                            {
                                "offset": 1,
                                "infons": {
                                    "section_type": "RESULTS",
                                    "type": "paragraph",
                                    "section": "Results",
                                },
                                "text": "Results text",
                            }
                        ],
                    }
                ]
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                extract.fetch_bioc_json = lambda *args, **kwargs: (payload, None)
                with contextlib.redirect_stdout(io.StringIO()):
                    passage_file = ensure_default_passage_file("12345")

                saved = json.loads(passage_file.read_text(encoding="utf-8"))
                self.assertEqual(passage_file, Path("data") / "passages" / "12345.json")
                self.assertEqual(saved["pmid"], "12345")
                self.assertEqual(saved["passage_count"], 1)
                self.assertEqual(saved["combined_results_text"], "Results text")
            finally:
                extract.fetch_bioc_json = original_fetch
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
