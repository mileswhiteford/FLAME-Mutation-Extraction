import unittest

from batch_pmid import (
    extract_results_passages,
    normalize_pmid_token,
    ordered_unique_pmids,
    split_pmid_cell,
)
from extract import is_content_block_error, optional_float_config


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
                        ],
                    }
                ]
            }
        ]

        extracted = extract_results_passages("23750226", payload)

        self.assertEqual(extracted["passage_count"], 1)
        self.assertEqual(extracted["methods_passage_count"], 1)
        self.assertEqual(extracted["passages"][0]["text"], "Results text")
        self.assertEqual(extracted["methods_passages"][0]["text"], "Methods text")
        self.assertEqual(extracted["combined_results_text"], "Results text")
        self.assertEqual(extracted["combined_methods_text"], "Methods text")


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


if __name__ == "__main__":
    unittest.main()
