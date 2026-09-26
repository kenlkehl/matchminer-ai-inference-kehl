"""Synthetic sources only: no NCCN text, patient records or endpoint access."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.trials import _guideline_prompts as prompts
from matchminer_ai.trials._guideline_audit import audit_accepted_response, audit_catalog
from matchminer_ai.trials._guideline_canonical import (
    CATALOG,
    LEAN_STATE,
    SELECTION,
    build_call,
    clinical_candidate,
    consolidate,
    partition_calls,
    selected_states,
    validate_catalog,
)
from matchminer_ai.trials._guideline_canonical import TASK as CATALOG_TASK
from matchminer_ai.trials._guideline_context import ContextBudgetError, pack_messages
from matchminer_ai.trials._guideline_details import build_detail_call, candidate_menus
from matchminer_ai.trials._guideline_generation import Client, read_chat_stream
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredConfig as LLMConfig,
    parse_model_json,
)
from matchminer_ai.trials._guideline_ownership import branch_ledger, page_owners
from matchminer_ai.trials._guideline_pipeline import run_guideline
from matchminer_ai.trials._guideline_repairs import TASK as REPAIR_TASK
from matchminer_ai.trials._guideline_repairs import (
    apply_repair,
    repair_response,
    validation_targets,
)
from matchminer_ai.trials._guideline_schema import (
    CANONICAL,
    DETAIL,
    EXTRACTION,
    FIELDS,
    format_space,
    materialize_evidence,
    normalize_evidence_lists,
    normalize_identical_groups,
    validate_canonical,
    validate_extraction,
    validate_shape,
    validate_state,
)
from matchminer_ai.trials._guideline_sources import (
    InputNotReady,
    inventory,
    load_guideline,
    packets,
    safe_child,
    select_context,
)
from matchminer_ai.trials._guideline_specificity import (
    explicit_lines,
    validate_decision_fields,
)

TEXT = "Fictional disease state alpha. Consider synthetic test A when finding B is present."


def state():
    fields = dict.fromkeys(FIELDS, "NA")
    fields.update(
        cancer_type_allowed="Fictional disease", cancer_burden_allowed="State alpha"
    )
    evidence = [{"page_id": "p0002", "line_ids": [1]}]
    return {
        "name": "Fictional alpha",
        "space": fields,
        "evidence": evidence,
        "diagnostic_workup": [
            {
                "name": "Synthetic test A",
                "conditions": "When finding B is present",
                "category": "Not specified",
                "evidence": [{"page_id": "p0002", "line_ids": [1]}],
            }
        ],
        "treatment_options": [],
        "uncertainties": ["No treatment options in supplied synthetic text"],
    }


def quoted_state():
    value = state()
    owners = [value, *value["diagnostic_workup"], *value["treatment_options"]]
    for owner in owners:
        owner["evidence"] = [{"page_id": "p0002", "source_text": TEXT}]
    return value


def extraction():
    return {
        "candidates": [
            {**state(), "defining_branch": {"page_id": "p0002", "line_ids": [1]}}
        ],
        "page_coverage": [
            {
                "page_id": "p0002",
                "disposition": "states_extracted",
                "reason": "Explicit synthetic state",
            }
        ],
        "uncertainties": [],
    }


def canonical():
    return {
        "groups": [
            {
                "candidate_ids": ["extract-0001-c001"],
                "name": "Fictional alpha",
                "space": state()["space"],
                "rationale": "Single state",
            }
        ],
        "context_only_candidates": [],
        "uncertainties": [],
    }


def catalog():
    return {
        "states": [{k: state()[k] for k in LEAN_STATE["properties"]}],
        "context_only_topics": [],
        "uncertainties": [],
    }


def make_library(base, page_text=TEXT):
    root = base / "source" / "markdown"
    folder = root / "fictional"
    folder.mkdir(parents=True)
    pdf = b"SYNTHETIC PDF PLACEHOLDER - no clinical data"
    (root.parent / "fictional.pdf").write_bytes(pdf)
    records, generated = [], {}
    for number, section, text in [
        (1, "00-front-matter", "Synthetic cover page"),
        (2, "10-guideline-sections/alpha", page_text),
    ]:
        relative = f"{section}/page-{number:04d}.md"
        page = folder / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# Synthetic page\n\n```text\n{text}\n```\n\n## Links\n".encode()
        page.write_bytes(raw)
        generated[relative] = digest(raw)
        records.append(
            {
                "pdf_page": number,
                "path": relative,
                "label": f"SYN-{number}",
                "internal_link_pages": [],
            }
        )
    atomic_json(
        folder / "manifest.json",
        {
            "generator": "nccn-pdf-to-markdown",
            "title": "Fictional",
            "version": "1.0",
            "page_count": 2,
            "source_pdf": "fictional.pdf",
            "source_sha256": digest(pdf),
            "pages": records,
            "generated_files": generated,
        },
    )
    atomic_json(
        root / "manifest.json",
        {
            "guidelines": [
                {
                    "folder": "fictional",
                    "title": "Fictional",
                    "version": "1.0",
                    "pages": 2,
                }
            ]
        },
    )
    return root


class LibraryFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = make_library(self.base)


class SourceTests(LibraryFixture, unittest.TestCase):
    def test_exact_copy_and_page_coverage(self):
        g = load_guideline(self.root, "fictional")
        self.assertEqual(list(g.pages), ["p0001", "p0002"])
        self.assertEqual(g.primary[0].text, TEXT)
        self.assertEqual(len(g.primary), 1)
        self.assertEqual(inventory(self.root)[1][0]["status"], "present_unverified")

    def test_partial_markdown_copy_rejected(self):
        path = self.root / "fictional/10-guideline-sections/alpha/page-0002.md"
        path.write_text("partially copied")
        with self.assertRaisesRegex(InputNotReady, "not fully copied"):
            load_guideline(self.root, "fictional")

    def test_missing_page_is_pending(self):
        (self.root / "fictional/10-guideline-sections/alpha/page-0002.md").unlink()
        self.assertEqual(inventory(self.root)[1][0]["status"], "pending_copy")
        with self.assertRaises(InputNotReady):
            load_guideline(self.root, "fictional")

    def test_partial_pdf_rejected(self):
        (self.root.parent / "fictional.pdf").write_bytes(b"partial")
        with self.assertRaisesRegex(InputNotReady, "Source PDF"):
            load_guideline(self.root, "fictional")

    def test_traversal_rejected(self):
        with self.assertRaises(InputNotReady):
            safe_child(self.root, "../../escape")

    def test_duplicate_page_number_rejected(self):
        path = self.root / "fictional/manifest.json"
        data = read_json(path)
        data["pages"][1]["pdf_page"] = 1
        atomic_json(path, data)
        with self.assertRaisesRegex(InputNotReady, "exactly once"):
            load_guideline(self.root, "fictional")

    def test_page_never_truncated(self):
        g = load_guideline(self.root, "fictional")
        with self.assertRaisesRegex(ValueError, "never cut"):
            list(packets(g, max_chars=10))
        with self.assertRaisesRegex(ValueError, "Required evidence"):
            select_context(g, {"p0002"}, "alpha", 10)
        context, omitted = select_context(g, {"p0002"}, "alpha", 1000)
        self.assertEqual(context[0].text, TEXT)
        self.assertEqual(omitted, [])


class SchemaTests(LibraryFixture, unittest.TestCase):
    def test_shape_feedback_identifies_missing_and_misplaced_fields_without_rewriting(
        self,
    ):
        missing = state()
        del missing["evidence"]
        original = copy.deepcopy(missing)
        with self.assertRaises(ValueError) as caught:
            validate_shape(missing, DETAIL)
        self.assertIn("missing keys=['evidence']", str(caught.exception))
        self.assertEqual(missing, original)

        misplaced = state()
        misplaced["space"]["evidence"] = copy.deepcopy(misplaced["evidence"])
        misplaced["space"]["uncertainties"] = copy.deepcopy(misplaced["uncertainties"])
        original = copy.deepcopy(misplaced)
        with self.assertRaises(ValueError) as caught:
            validate_shape(misplaced, DETAIL)
        self.assertIn("response.space:", str(caught.exception))
        self.assertIn(
            "unexpected keys=['evidence', 'uncertainties']", str(caught.exception)
        )
        self.assertEqual(misplaced, original)

    def test_trialspace_order_and_single_line(self):
        text = format_space(state()["space"])
        positions = [text.index(label + ":") for label in FIELDS.values()]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("\n", text)

    def test_nonexistent_lines_and_page_ids_rejected(self):
        g = load_guideline(self.root, "fictional")
        for field, value in [
            ("line_ids", [999]),
            ("line_ids", [0]),
            ("line_ids", [True]),
            ("page_id", "p9999"),
        ]:
            sample = state()
            sample["diagnostic_workup"][0]["evidence"][0][field] = value
            with self.assertRaises(ValueError):
                validate_state(sample, g.pages)

    def test_evidence_text_is_copied_by_code(self):
        result = materialize_evidence(
            state(), load_guideline(self.root, "fictional").pages
        )
        self.assertEqual(result["evidence"][0]["quote"], TEXT)
        self.assertEqual(result["evidence"][0]["pdf_page"], 2)
        self.assertEqual(
            result["evidence"][0]["source_lines"], [{"line": 1, "text": TEXT}]
        )

    def test_cited_passage_may_include_blank_lines_but_not_only_blanks(self):
        pages = load_guideline(self.root, "fictional").pages
        pages["p0002"] = replace(
            pages["p0002"], text=TEXT + "\n\nAdditional synthetic text"
        )
        sample = state()
        sample["evidence"][0]["line_ids"] = [1, 2, 3]
        validate_state(sample, pages)
        sample["evidence"][0]["line_ids"] = [2]
        with self.assertRaisesRegex(ValueError, "entirely of blank"):
            validate_state(sample, pages)

    def test_missing_evidence_rejected(self):
        sample = state()
        sample["evidence"] = []
        with self.assertRaises(ValueError):
            validate_state(sample, load_guideline(self.root, "fictional").pages)

    def test_citation_normalization_preserves_references_and_clinical_fields(self):
        pages = load_guideline(self.root, "fictional").pages
        pages["p0002"] = replace(pages["p0002"], text="\n".join([TEXT] * 20))
        sample = state()
        sample["evidence"][0]["line_ids"] = list(range(1, 16)) + [2, 3]
        before = copy.deepcopy(sample)
        self.assertEqual(normalize_evidence_lists(sample), 1)
        self.assertEqual(
            [i for e in sample["evidence"] for i in e["line_ids"]], list(range(1, 16))
        )
        self.assertEqual(
            {k: v for k, v in sample.items() if k != "evidence"},
            {k: v for k, v in before.items() if k != "evidence"},
        )
        validate_state(sample, pages)
        sample["evidence"][0]["line_ids"] = [True, -1]
        self.assertEqual(normalize_evidence_lists(sample), 0)
        with self.assertRaises(ValueError):
            validate_state(sample, pages)

    def test_coverage_cannot_omit_page(self):
        sample = extraction()
        sample["page_coverage"] = []
        with self.assertRaisesRegex(ValueError, "every PRIMARY"):
            validate_extraction(
                sample, load_guideline(self.root, "fictional").pages, ["p0002"]
            )

    def test_coverage_accepts_option_evidence_on_primary_page(self):
        sample = extraction()
        sample["candidates"][0]["evidence"] = [{"page_id": "p0001", "line_ids": [1]}]
        validate_extraction(
            sample, load_guideline(self.root, "fictional").pages, ["p0002"]
        )

    def test_context_only_branch_cannot_be_extracted_by_another_pages_owner(self):
        sample = extraction()
        sample["candidates"][0]["defining_branch"]["page_id"] = "p0001"
        original = copy.deepcopy(sample)
        with self.assertRaisesRegex(ValueError, "CONTEXT-only branch belongs"):
            validate_extraction(
                sample, load_guideline(self.root, "fictional").pages, ["p0002"]
            )
        self.assertEqual(sample, original)

    def test_background_disposition_cannot_claim_a_branch_even_with_valid_primary_citation(
        self,
    ):
        sample = extraction()
        sample["page_coverage"][0]["disposition"] = "context_only"
        with self.assertRaisesRegex(ValueError, "only discussion/routing"):
            validate_extraction(
                sample, load_guideline(self.root, "fictional").pages, ["p0002"]
            )
        sample["candidates"] = []
        validate_extraction(
            sample, load_guideline(self.root, "fictional").pages, ["p0002"]
        )

    def test_invalid_branch_address_cannot_be_reassigned_by_citation_repair(self):
        sample = extraction()
        sample["candidates"][0]["defining_branch"]["line_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "Extraction ownership: invalid"):
            validate_extraction(
                sample, load_guideline(self.root, "fictional").pages, ["p0002"]
            )

    def test_legacy_extractions_remain_auditable_only_with_explicit_legacy_contract(
        self,
    ):
        sample = extraction()
        del sample["candidates"][0]["defining_branch"]
        pages = load_guideline(self.root, "fictional").pages
        with self.assertRaisesRegex(ValueError, "defining_branch"):
            validate_extraction(sample, pages, ["p0002"])
        validate_extraction(sample, pages, ["p0002"], require_ownership=False)

    def test_canonicalization_cannot_drop_candidates_or_duplicate_states(self):
        with self.assertRaisesRegex(ValueError, "Account for all"):
            validate_canonical(canonical(), ["extract-0001-c001", "missing"])
        sample = canonical()
        sample["groups"].append(copy.deepcopy(sample["groups"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            validate_canonical(sample, ["extract-0001-c001"])

    def test_broad_candidate_can_support_distinct_states_with_context_accounted(self):
        sample = canonical()
        other = copy.deepcopy(sample["groups"][0])
        other["space"]["cancer_burden_allowed"] = "Synthetic distinct state beta"
        sample["groups"].append(other)
        sample["context_only_candidates"] = [
            {"candidate_id": "context", "reason": "General context"}
        ]
        validate_canonical(sample, ["extract-0001-c001", "context"])


class OwnershipTests(unittest.TestCase):
    def test_page_assignments_are_a_partition(self):
        self.assertEqual(
            page_owners({"first": ["p1"], "second": ["p2"]}, ["p1", "p2"]),
            {"p1": "first", "p2": "second"},
        )
        with self.assertRaisesRegex(ValueError, "multiple calls"):
            page_owners({"first": ["p1"], "second": ["p1", "p2"]}, ["p1", "p2"])
        with self.assertRaisesRegex(ValueError, "cover the guideline exactly"):
            page_owners({"first": ["p1"]}, ["p1", "p2"])

    def test_split_alternatives_share_one_branch_owner(self):
        result = extraction()
        second = copy.deepcopy(result["candidates"][0])
        second["space"]["prior_treatment_required"] = "Prior synthetic therapy"
        result["candidates"].append(second)
        packets = {"extract-0001": {"primary_page_ids": ["p0002"], "result": result}}
        ledger = branch_ledger(packets, ["p0002"])
        self.assertEqual(len(ledger["branches"]), 1)
        self.assertEqual(
            ledger["branches"][0]["candidate_ids"],
            ["extract-0001-c001", "extract-0001-c002"],
        )

    def test_background_reextraction_fails_ledger_even_when_page_assignment_is_disjoint(
        self,
    ):
        packets = {
            "extract-0001": {"primary_page_ids": ["p0002"], "result": extraction()},
            "extract-0002": {"primary_page_ids": ["p0003"], "result": extraction()},
        }
        with self.assertRaisesRegex(ValueError, "another page's owner"):
            branch_ledger(packets, ["p0002", "p0003"])


class SpecificityTests(unittest.TestCase):
    def test_mixed_boolean_conditions_require_grouping_without_automatic_rewriting(
        self,
    ):
        for field, criterion in (
            (
                "cancer_burden_allowed",
                "Recurrent (local or regional) OR metastatic AND visceral crisis",
            ),
            ("biomarkers_required", "(Marker-A OR Marker-B AND Marker-C) AND Marker-D"),
            ("prior_treatment_required", "Agent A AND Agent B OR Agent C"),
        ):
            with self.subTest(field=field):
                value = state()
                value["space"][field] = criterion
                original = copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError, "same parenthesis level"):
                    validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_nested_grouping_and_single_operator_lists_preserve_clinical_text(self):
        for criterion in (
            "(Marker-A OR Marker-B) AND Marker-C",
            "Marker-A OR (Marker-B AND Marker-C)",
            "((Marker-A OR Marker-B) AND Marker-C) OR Marker-D",
            "Marker-A OR Marker-B",
            "Marker-A AND Marker-B",
        ):
            with self.subTest(criterion=criterion):
                value = state()
                value["space"]["biomarkers_required"] = criterion
                original = copy.deepcopy(value)
                validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_visceral_crisis_alternative_is_rejected_in_prior_fields_without_rewriting(
        self,
    ):
        for field in ("prior_treatment_required", "prior_treatment_excluded"):
            for criterion in (
                "Visceral crisis",
                "Visceral crisis OR endocrine refractory",
                "Endocrine refractory OR (visceral crisis)",
                "Without visceral crisis",
            ):
                with self.subTest(field=field, criterion=criterion):
                    value = state()
                    value["space"][field] = criterion
                    original = copy.deepcopy(value)
                    with self.assertRaisesRegex(ValueError, "change OR to AND"):
                        validate_decision_fields(value, "Fictional guideline")
                    self.assertEqual(value, original)

    def test_burden_and_treatment_response_alternatives_remain_separate(self):
        crisis, refractory = state(), state()
        crisis["space"]["cancer_burden_allowed"] = "Metastatic; visceral crisis"
        refractory["space"]["cancer_burden_allowed"] = "Metastatic"
        refractory["space"]["prior_treatment_required"] = "Endocrine refractory"
        for value in (crisis, refractory):
            value["name"] = "Fictional disease, first-line cytotoxic therapy"
            value["space"]["prior_treatment_excluded"] = (
                "Cytotoxic therapy for metastatic disease"
            )
            original = copy.deepcopy(value)
            validate_decision_fields(value, "Fictional guideline")
            self.assertEqual(value, original)
        self.assertEqual(crisis["space"]["prior_treatment_required"], "NA")
        self.assertNotIn("visceral", refractory["space"]["cancer_burden_allowed"])
        self.assertNotEqual(
            format_space(crisis["space"]), format_space(refractory["space"])
        )

    def test_crisis_mentioned_as_context_of_actual_prior_therapy_is_not_rejected(self):
        value = state()
        value["space"]["prior_treatment_required"] = (
            "Prior chemotherapy for visceral crisis"
        )
        original = copy.deepcopy(value)
        validate_decision_fields(value, "Fictional guideline")
        self.assertEqual(value, original)

    def test_unrestricted_line_names_do_not_invent_prior_treatment_requirements(self):
        for label in (
            "first-line or subsequent therapy",
            "1st-line and any later therapy",
            "1L or subsequent treatment",
            "First–line or later lines",
        ):
            with self.subTest(label=label):
                value = state()
                value["name"] = "Fictional disease, " + label
                original = copy.deepcopy(value)
                validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_unrestricted_name_exception_does_not_mask_a_separate_line_boundary(self):
        for label in (
            "second-line or subsequent therapy",
            "first-line or second-line therapy",
            "first-line or subsequent therapy; second-line subgroup",
        ):
            with self.subTest(label=label):
                value = state()
                value["name"] = "Fictional disease, " + label
                with self.assertRaisesRegex(ValueError, "only in the name"):
                    validate_decision_fields(value, "Fictional guideline")

    def test_unrestricted_line_wording_still_does_not_belong_in_burden(self):
        value = state()
        value["name"] = "Fictional disease, first-line or subsequent therapy"
        value["space"]["cancer_burden_allowed"] = (
            "Metastatic; first-line or subsequent therapy"
        )
        with self.assertRaisesRegex(
            ValueError, "burden_allowed contains treatment-line"
        ):
            validate_decision_fields(value, "Fictional guideline")

    def test_line_only_in_name_is_rejected_without_modifying_the_state(self):
        value = state()
        value["name"] = "Fictional metastatic disease, first-line treatment"
        original = copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError, "only in the name") as raised:
            validate_decision_fields(value, "Fictional guideline")
        self.assertIn(value["name"], str(raised.exception))
        self.assertEqual(value, original)

    def test_shared_source_line_bin_and_ordinal_spellings_are_preserved(self):
        value = state()
        value["name"] = "Fictional disease, second-line or third-line treatment"
        value["space"]["cancer_burden_allowed"] = "Metastatic"
        value["space"]["prior_treatment_required"] = (
            "One OR two prior lines of systemic therapy for metastatic disease"
        )
        original = copy.deepcopy(value)
        validate_decision_fields(value, "Fictional")
        self.assertEqual(value, original)
        self.assertEqual(explicit_lines("2nd/3rd-line systemic treatment"), {"2", "3"})
        self.assertEqual(
            explicit_lines("Fourth–line and beyond; prior 1L therapy"), {"1", "4"}
        )
        self.assertEqual(
            explicit_lines("Histology grade 3; stage IV; line of treatment unknown"),
            set(),
        )

    def test_treatment_lines_are_rejected_in_burden_without_rewriting(self):
        for burden in (
            "Metastatic; second-line",
            "Stage IV; 2L",
            "Metastatic; exactly one prior line of chemotherapy",
        ):
            with self.subTest(burden=burden):
                value = state()
                value["space"]["cancer_burden_allowed"] = burden
                value["space"]["prior_treatment_required"] = (
                    "Exactly one prior line of chemotherapy for metastatic disease"
                )
                original = copy.deepcopy(value)
                with self.assertRaisesRegex(
                    ValueError, "burden_allowed contains treatment-line"
                ):
                    validate_decision_fields(value, "Fictional")
                self.assertEqual(value, original)

    def test_first_line_excludes_treatment_in_source_setting_without_double_negation(
        self,
    ):
        value = state()
        value["name"] = "Fictional metastatic disease, first-line chemotherapy"
        value["space"]["prior_treatment_excluded"] = (
            "Chemotherapy for metastatic disease"
        )
        validate_decision_fields(value, "Fictional")
        value["space"]["prior_treatment_excluded"] = (
            "No prior chemotherapy for metastatic disease"
        )
        with self.assertRaisesRegex(ValueError, "reverses exclusion logic"):
            validate_decision_fields(value, "Fictional")

    def test_generic_breast_hr_criteria_require_model_correction(self):
        for marker in (
            "HR-positive",
            "HR negative",
            "hormone receptor positive",
            "HR+",
            "HR−",
        ):
            with self.subTest(marker=marker):
                value = state()
                value["space"]["biomarkers_required"] = marker
                with self.assertRaisesRegex(ValueError, "ER/PR"):
                    validate_decision_fields(value, "Synthetic Breast Cancer Guideline")
                validate_decision_fields(value, "Other fictional disease")

    def test_explicit_receptors_discordance_and_hrd_are_not_rewritten(self):
        for marker in (
            "(ER-positive OR PR-positive) AND HER2-negative",
            "ER-negative AND PR-negative AND HER2-negative",
            "ER-positive AND PR-negative",
            "HRD-negative",
        ):
            value = state()
            value["space"]["biomarkers_required"] = marker
            original = copy.deepcopy(value)
            validate_decision_fields(value, "Breast Cancer")
            self.assertEqual(value, original)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = Client(
            LLMConfig(model="synthetic", attempts=2, tokenizer_mode="bytes"),
            Path(self.temp.name),
        )
        self.messages = [{"role": "user", "content": "Return synthetic JSON"}]

    def response(self, finish="stop", content=None):
        return {
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": content or json.dumps(canonical())},
                }
            ]
        }

    def validate(self, value):
        validate_canonical(value, ["extract-0001-c001"])

    def test_resume_makes_no_network_call(self):
        with patch.object(self.client, "_http", return_value=self.response()) as http:
            self.client.complete("job", self.messages, CANONICAL, self.validate)
            self.client.complete("job", self.messages, CANONICAL, self.validate)
            self.assertEqual(http.call_count, 1)

    def test_extraction_retry_preserves_recent_errors_on_resume(self):
        def reject(value):
            raise ValueError(value["error"])

        with (
            patch.object(self.client, "_http", side_effect=[
                self.response(content=json.dumps({"error": "first field problem"})),
                self.response(content=json.dumps({"error": "second field problem"})),
            ]),
            patch("matchminer_ai.llm.structured.time.sleep"),
            self.assertRaises(EndpointError),
        ):
            self.client.complete("history", self.messages, EXTRACTION, reject)

        def respond(endpoint, body, **kwargs):
            feedback = body["messages"][-1]["content"]
            self.assertIn("first field problem", feedback)
            self.assertIn("second field problem", feedback)
            return self.response(content=json.dumps({"valid": True}))

        def validate(value):
            if "error" in value:
                reject(value)

        with patch.object(self.client, "_http", side_effect=respond):
            self.assertEqual(
                self.client.complete("history", self.messages, EXTRACTION, validate),
                {"valid": True},
            )

    def test_resume_recovers_response_saved_before_acceptance(self):
        with patch.object(self.client, "_http", return_value=self.response()):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        next(Path(self.temp.name).rglob("accepted.json")).unlink()
        with patch.object(
            self.client, "_http", side_effect=AssertionError("No network call expected")
        ):
            result = self.client.complete(
                "job", self.messages, CANONICAL, self.validate
            )
        self.assertEqual(result, canonical())

    def test_resume_revalidates_numeric_threshold_without_regeneration_or_edits(self):
        self.client.config = replace(self.client.config, attempts=1)
        value = state()
        value["space"]["cancer_burden_allowed"] = "(A OR B) AND score 2 or higher"
        response = self.response(content=json.dumps(value))

        def old_validator(result):
            raise ValueError("Old checker misclassified the numeric threshold as OR")

        with patch.object(self.client, "_http", return_value=response):
            with self.assertRaisesRegex(EndpointError, "numeric threshold"):
                self.client.complete("job", self.messages, DETAIL, old_validator)
        path = next(Path(self.temp.name).rglob("attempt-1.json"))
        original = path.read_bytes()
        with patch.object(
            self.client, "_http", side_effect=AssertionError("No regeneration expected")
        ):
            result = self.client.complete(
                "job",
                self.messages,
                DETAIL,
                lambda result: validate_decision_fields(result, "Fictional guideline"),
            )
        self.assertEqual(result, value)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(
            len(list(Path(self.temp.name).rglob("request-attempt-*.json"))), 1
        )
        self.assertEqual(
            read_json(path.with_name("accepted.json"))["response_sha256"],
            digest(response),
        )

    def test_resume_preserves_interrupted_request_without_response(self):
        with patch.object(self.client, "_http", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.client.complete("job", self.messages, CANONICAL, self.validate)
        original = next(Path(self.temp.name).rglob("request-attempt-1.json"))
        saved = original.read_bytes()
        with patch.object(self.client, "_http", return_value=self.response()):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        self.assertEqual(original.read_bytes(), saved)
        self.assertTrue(original.with_name("request-attempt-2.json").exists())
        self.assertTrue(original.with_name("attempt-2.json").exists())
        self.assertFalse(original.with_name("attempt-1.json").exists())

    def test_resume_uses_saved_validation_failure_in_first_new_attempt(self):
        self.client.config = replace(self.client.config, attempts=1)
        with patch.object(
            self.client, "_http", return_value=self.response(content='{"bad":1}')
        ):
            with self.assertRaises(EndpointError):
                self.client.complete("job", self.messages, CANONICAL, self.validate)
        with patch.object(self.client, "_http", return_value=self.response()) as http:
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        self.assertIn(
            "failed validation", http.call_args.args[1]["messages"][-1]["content"]
        )
        self.assertTrue(list(Path(self.temp.name).rglob("attempt-1.json")))
        self.assertTrue(list(Path(self.temp.name).rglob("attempt-2.json")))

    def test_token_limited_valid_json_is_rejected(self):
        with (
            patch.object(self.client, "_http", return_value=self.response("length")),
            patch("time.sleep"),
        ):
            with self.assertRaisesRegex(EndpointError, "finish_reason='length'"):
                self.client.complete("job", self.messages, CANONICAL, self.validate)
        self.assertFalse(list(Path(self.temp.name).rglob("accepted.json")))

    def test_schema_retry_feedback(self):
        with (
            patch.object(
                self.client,
                "_http",
                side_effect=[self.response(content='{"bad": 1}'), self.response()],
            ) as http,
            patch("time.sleep"),
        ):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
            self.assertIn(
                "failed validation",
                http.call_args_list[1].args[1]["messages"][-1]["content"],
            )

    def test_conflicting_scalar_keys_are_rejected_instead_of_silently_overwritten(self):
        duplicate = json.dumps(canonical()).replace(
            '"Single state"', '"Single state", "rationale": "Conflicting meaning"'
        )
        with (
            patch.object(
                self.client,
                "_http",
                side_effect=[self.response(content=duplicate), self.response()],
            ) as http,
            patch("time.sleep"),
        ):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        self.assertEqual(http.call_count, 2)
        self.assertIn(
            "Conflicting duplicate JSON key",
            http.call_args.args[1]["messages"][-1]["content"],
        )

    def test_duplicate_lists_preserve_all_distinct_model_generated_items(self):
        content = '{"options": [{"name": "Synthetic A"}], "options": [{"name": "Synthetic B"}], "options": [{"name": "Synthetic A"}], "category": "Synthetic", "category": "Synthetic"}'
        value, counts = parse_model_json(content)
        self.assertEqual(
            value["options"], [{"name": "Synthetic A"}, {"name": "Synthetic B"}]
        )
        self.assertEqual(
            counts,
            {
                "identical_duplicate_keys": 1,
                "merged_list_keys": 2,
                "recovered_list_items": 1,
            },
        )

    def test_legacy_cache_is_reparsed_without_generating_clinical_content(self):
        with patch.object(self.client, "_http", return_value=self.response()):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        path = next(Path(self.temp.name).rglob("accepted.json"))
        value = read_json(path)
        del value["json_normalization_version"]
        atomic_json(path, value)
        with patch.object(
            self.client, "_http", side_effect=AssertionError("No generation expected")
        ):
            self.assertEqual(
                self.client.complete("job", self.messages, CANONICAL, self.validate),
                canonical(),
            )
        self.assertIn("json_normalization_version", read_json(path))

    def test_parser_upgrade_keeps_validated_citation_repairs_when_clinical_content_matches(
        self,
    ):
        original = state()
        original["evidence"][0]["line_ids"] = [999]
        with patch.object(
            self.client,
            "_http",
            return_value=self.response(content=json.dumps(original)),
        ):
            self.client.complete("job", self.messages, DETAIL, lambda value: None)
        path = next(Path(self.temp.name).rglob("accepted.json"))
        accepted = read_json(path)
        accepted["result"]["evidence"][0]["line_ids"] = [1]
        accepted["result_sha256"] = digest(accepted["result"])
        accepted["repair_applied"] = True
        del accepted["json_normalization_version"]
        atomic_json(path, accepted)
        with patch.object(
            self.client,
            "_http",
            side_effect=AssertionError("No repeated repair expected"),
        ):
            result = self.client.complete(
                "job", self.messages, DETAIL, lambda value: None
            )
        self.assertEqual(result["evidence"][0]["line_ids"], [1])
        audit_accepted_response(path.parent, read_json(path))

    def test_parser_upgrade_restores_previously_discarded_options_even_after_citation_repair(
        self,
    ):
        original = state()
        option = copy.deepcopy(original["diagnostic_workup"][0])
        option["name"] = "Synthetic additional option"
        raw = (
            json.dumps(original)[:-1]
            + ', "diagnostic_workup": ['
            + json.dumps(option)
            + "]}"
        )
        with patch.object(
            self.client, "_http", return_value=self.response(content=raw)
        ):
            self.client.complete("job", self.messages, DETAIL, lambda value: None)
        path = next(Path(self.temp.name).rglob("accepted.json"))
        accepted = read_json(path)
        accepted["result"]["diagnostic_workup"] = [option]
        accepted["result_sha256"] = digest(accepted["result"])
        accepted["repair_applied"] = True
        del accepted["json_normalization_version"]
        atomic_json(path, accepted)
        with patch.object(
            self.client,
            "_http",
            side_effect=AssertionError("No clinical generation expected"),
        ):
            result = self.client.complete(
                "job", self.messages, DETAIL, lambda value: None
            )
        self.assertEqual(len(result["diagnostic_workup"]), 2)
        audit_accepted_response(path.parent, read_json(path))

    def test_raw_response_audit_detects_changed_clinical_content_even_with_new_checksum(
        self,
    ):
        with patch.object(self.client, "_http", return_value=self.response()):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        path = next(Path(self.temp.name).rglob("accepted.json"))
        accepted = read_json(path)
        audit_accepted_response(path.parent, accepted)
        accepted["result"]["uncertainties"].append("Synthetic silently changed content")
        accepted["result_sha256"] = digest(accepted["result"])
        with self.assertRaisesRegex(ValueError, "dropped or changed provider content"):
            audit_accepted_response(path.parent, accepted)

    def test_corrupt_cache_rejected(self):
        with patch.object(self.client, "_http", return_value=self.response()):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        path = next(Path(self.temp.name).rglob("accepted.json"))
        data = read_json(path)
        data["result"]["groups"][0]["name"] = "changed"
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "Corrupted"):
            self.client.complete("job", self.messages, CANONICAL, self.validate)

    def test_secret_not_serialized_and_google_sampling_reasoning_defaults(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-secret"}),
            patch.object(self.client, "_http", return_value=self.response()) as http,
        ):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
            body = http.call_args.args[1]
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(
                (body["temperature"], body["top_p"], body["top_k"]), (1.0, 0.95, 64)
            )
            self.assertEqual(body["max_tokens"], 100000)
        for path in Path(self.temp.name).rglob("*.json"):
            self.assertNotIn("synthetic-secret", path.read_text())

    def test_generic_endpoint_extensions_can_be_omitted_explicitly(self):
        self.client.config = replace(self.client.config, thinking="default", top_k=0)
        with patch.object(self.client, "_http", return_value=self.response()) as http:
            self.client.complete("job", self.messages, CANONICAL, self.validate)
            self.assertNotIn("chat_template_kwargs", http.call_args.args[1])
            self.assertNotIn("top_k", http.call_args.args[1])

    def test_endpoint_token_count_uses_same_thinking_template(self):
        self.client.config = replace(self.client.config, tokenizer_mode="endpoint")
        with patch.object(
            self.client, "_http", return_value={"count": 123, "max_model_len": 262144}
        ) as http:
            self.assertEqual(self.client.count_tokens(self.messages), 123)
            self.assertEqual(http.call_args.args[0], "/tokenize")
            self.assertEqual(
                http.call_args.args[1]["chat_template_kwargs"],
                {"enable_thinking": True},
            )
            self.assertTrue(http.call_args.kwargs["server_root"])

    def test_output_reserve_is_never_squeezed_to_fit_prompt(self):
        self.client.config = replace(self.client.config, context_window=102060)
        with patch.object(
            self.client, "_http", side_effect=AssertionError("No generation expected")
        ):
            with self.assertRaisesRegex(ValueError, "reserving 100000"):
                self.client.complete("job", self.messages, CANONICAL, self.validate)

    def test_retry_uses_safety_space_without_reducing_output_reserve(self):
        count = self.client.count_tokens(self.messages)
        self.client.config = replace(
            self.client.config, context_window=100000 + 2048 + count
        )
        with (
            patch.object(
                self.client,
                "_http",
                side_effect=[self.response(content='{"bad":1}'), self.response()],
            ) as http,
            patch("time.sleep"),
        ):
            self.client.complete("job", self.messages, CANONICAL, self.validate)
        retry = http.call_args_list[1].args[1]
        self.assertGreater(
            self.client.count_tokens(retry["messages"]), self.client.prompt_budget
        )
        self.assertLessEqual(
            self.client.count_tokens(retry["messages"]) + retry["max_tokens"],
            self.client.config.context_window,
        )
        self.assertTrue(list(Path(self.temp.name).rglob("request-attempt-2.json")))


class StreamTests(unittest.TestCase):
    def events(self, deltas, finish="stop"):
        for delta in deltas:
            yield (
                "data: "
                + json.dumps(
                    {
                        "model": "synthetic",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                + "\n"
            ).encode()
        yield (
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            )
            + "\n"
        ).encode()
        yield b'data: {"choices": [], "usage": {"completion_tokens": 17}}\n'
        yield b"data: [DONE]\n"

    def test_stream_keeps_reasoning_content_finish_and_usage(self):
        value = read_chat_stream(
            self.events(
                [
                    {"reasoning": "Synthetic "},
                    {"reasoning_content": "reasoning"},
                    {"content": '{"ok":'},
                    {"content": "true}"},
                ]
            )
        )
        self.assertEqual(value["choices"][0]["message"]["content"], '{"ok":true}')
        self.assertEqual(
            value["choices"][0]["message"]["reasoning"], "Synthetic reasoning"
        )
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertEqual(value["usage"]["completion_tokens"], 17)

    def test_stream_aborts_padding_and_repeated_gibberish_without_accepting_it(self):
        variants = " ".join(a + b + c for a in "lLa" for b in "lde" for c in "aLl")
        for text in (" \n" * 5000, "l la L de " * 1500, (variants + " ") * 100):
            value = read_chat_stream(self.events([{"content": text}]))
            self.assertEqual(
                value["choices"][0]["finish_reason"], "client_aborted_repetition"
            )
            self.assertEqual(value["choices"][0]["message"]["content"], text)
            self.assertIn("abort_reason", value["transport"])

    def test_stream_does_not_cap_long_meaningful_output_or_accept_truncation(self):
        text = json.dumps(
            {
                "synthetic_items": [
                    f"Distinct fictional observation {i}" for i in range(2000)
                ]
            }
        )
        value = read_chat_stream(self.events([{"content": text}]))
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertEqual(value["choices"][0]["message"]["content"], text)
        self.assertEqual(
            read_chat_stream(self.events([{"content": text}], finish="length"))[
                "choices"
            ][0]["finish_reason"],
            "length",
        )
        incomplete = list(self.events([{"content": "{"}]))[:1]
        self.assertIsNone(read_chat_stream(incomplete)["choices"][0]["finish_reason"])

    def test_stream_detects_cycles_of_complete_clinical_definitions(self):
        spaces = []
        for i in range(40):
            item = copy.deepcopy(state()["space"])
            item["cancer_burden_allowed"] = f"Fictional distinct state {i}"
            spaces.append(item)

        def streamed(values):
            text = json.dumps(
                {"spaces": values, "context_only_topics": [], "uncertainties": []}
            )
            return read_chat_stream(
                self.events(
                    [{"content": text[i : i + 137]} for i in range(0, len(text), 137)]
                ),
                unique_spaces=True,
            )

        self.assertEqual(streamed(spaces)["choices"][0]["finish_reason"], "stop")
        looped = streamed(spaces * 4)
        self.assertEqual(
            looped["choices"][0]["finish_reason"], "client_aborted_repetition"
        )
        self.assertIn(
            "complete clinical definition", looped["transport"]["abort_reason"]
        )

    def test_stream_detects_repeated_options_across_repeated_json_keys(self):
        option = state()["diagnostic_workup"][0]
        text = "{" + ('"treatment_options": [' + json.dumps(option) + "],") * 50
        deltas = [{"content": text[i : i + 19]} for i in range(0, len(text), 19)]
        value = read_chat_stream(self.events(deltas), unique_options=True)
        self.assertEqual(
            value["choices"][0]["finish_reason"], "client_aborted_repetition"
        )
        self.assertIn("complete clinical option", value["transport"]["abort_reason"])
        self.assertLess(len(value["choices"][0]["message"]["content"]), len(text) // 2)
        # Multiple-state extraction may legitimately repeat an option; it does not
        # opt into this single-state guard.
        self.assertEqual(
            read_chat_stream(self.events(deltas))["choices"][0]["finish_reason"], "stop"
        )

    def test_stream_option_guard_preserves_distinct_conditions_and_escaped_strings(
        self,
    ):
        options = []
        for i in range(100):
            option = copy.deepcopy(state()["diagnostic_workup"][0])
            option["name"] = 'Fictional "test" with {braces} and \\ escaped text'
            option["conditions"] = f"Distinct synthetic condition {i}"
            options.append(option)
        text = json.dumps({"treatment_options": options})
        value = read_chat_stream(
            self.events([{"content": char} for char in text]), unique_options=True
        )
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertEqual(value["choices"][0]["message"]["content"], text)

    def test_stream_option_guard_counts_clinical_content_independently_of_citations(
        self,
    ):
        options = []
        for i in range(20):
            option = copy.deepcopy(state()["diagnostic_workup"][0])
            option["evidence"][0]["line_ids"] = [i + 1]
            options.append(option)
        value = read_chat_stream(
            self.events([{"content": json.dumps({"diagnostic_workup": options})}]),
            unique_options=True,
        )
        self.assertEqual(
            value["choices"][0]["finish_reason"], "client_aborted_repetition"
        )

    def test_scattered_repetitions_do_not_abort_a_catalog_still_adding_new_content(
        self,
    ):
        spaces = []
        options = []
        for i in range(100):
            space = copy.deepcopy(state()["space"])
            space["cancer_burden_allowed"] = f"Distinct synthetic state {i}"
            spaces.append(space)
            option = copy.deepcopy(state()["diagnostic_workup"][0])
            option["conditions"] = f"Distinct synthetic condition {i}"
            options.append(option)
            if i % 10 == 0:
                spaces.append(copy.deepcopy(spaces[0]))
                options.append(copy.deepcopy(options[0]))
        for value, flags in (
            ({"spaces": spaces}, {"unique_spaces": True}),
            ({"diagnostic_workup": options}, {"unique_options": True}),
        ):
            text = json.dumps(value)
            result = read_chat_stream(
                self.events(
                    [{"content": text[i : i + 211]} for i in range(0, len(text), 211)]
                ),
                **flags,
            )
            self.assertEqual(result["choices"][0]["finish_reason"], "stop")


class CanonicalBatchTests(LibraryFixture, unittest.TestCase):
    def test_consolidation_uses_all_candidate_evidence_without_unrelated_source_pages(
        self,
    ):
        g = load_guideline(self.root, "fictional")
        base = g.pages["p0002"]
        g.pages["p0003"] = replace(
            base, id="p0003", number=3, text="Unrelated synthetic population"
        )
        g.pages["p0004"] = replace(
            base, id="p0004", number=4, text="Synthetic option branch"
        )
        candidate = {"candidate_id": "opaque-secret", **state()}
        candidate["diagnostic_workup"][0]["evidence"] = [
            {"page_id": "p0004", "line_ids": [1]}
        ]
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        pages, omitted, messages = build_call(client, g, [candidate])
        self.assertEqual({p.id for p in pages}, {"p0002", "p0004"})
        self.assertIn("p0003", omitted)
        text = json.dumps(messages)
        self.assertIn("Synthetic option branch", text)
        self.assertNotIn("Unrelated synthetic population", text)
        self.assertNotIn("opaque-secret", text)
        self.assertEqual(client.config.max_tokens, 100000)

    def test_completed_prefix_requests_are_stable_when_later_candidates_arrive(self):
        candidates = [{"candidate_id": f"c{i}"} for i in range(250)]

        def pack(client, guideline, members, context_chars):
            return [], [], [{"role": "user", "content": json.dumps(members)}]

        with patch(
            "matchminer_ai.trials._guideline_canonical.build_call", side_effect=pack
        ):
            prefix = partition_calls(None, None, candidates[:200])
            expanded = partition_calls(None, None, candidates)
        self.assertEqual(prefix, expanded[: len(prefix)])
        self.assertEqual(sum(len(items) for items, packed in expanded), 250)

    def test_large_catalog_splits_and_selects_clinical_content_without_ids(self):
        guideline = load_guideline(self.root, "fictional")
        candidates = [
            {"candidate_id": f"opaque-secret-{i}", **state()} for i in range(1, 4)
        ]
        candidates[2]["name"] = "Fictional beta"
        candidates[2]["space"]["cancer_burden_allowed"] = "State beta"
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )

        def bounded(client, guideline, members, context_chars):
            if len(members) > 1:
                raise ContextBudgetError("Synthetic source budget exceeded")
            return build_call(client, guideline, members, context_chars)

        def complete(job, messages, schema, validator, repair_handler=None):
            rendered = json.dumps(messages)
            for forbidden in (
                "opaque-secret",
                "candidate_id",
                "group_id",
                "representative_id",
                "source_batch_numbers",
            ):
                self.assertNotIn(forbidden, rendered)
            if schema == SELECTION:
                self.assertNotIn(TEXT, rendered)
                self.assertIn("source_backed_definitions", rendered)
                value = {
                    "spaces": [candidates[0]["space"], candidates[2]["space"]],
                    "context_only_topics": [],
                    "uncertainties": [],
                }
            else:
                candidate = candidates[int(job.rsplit("-", 1)[-1]) - 1]
                value = {
                    "states": [{k: candidate[k] for k in LEAN_STATE["properties"]}],
                    "context_only_topics": [],
                    "uncertainties": [],
                }
            validator(value)
            return value

        def run_jobs(jobs, workers, function):
            return {key: function(key, data) for key, data in jobs}, {}

        output = self.base / "batched"
        with (
            patch(
                "matchminer_ai.trials._guideline_canonical.build_call",
                side_effect=bounded,
            ),
            patch.object(client, "complete", side_effect=complete),
        ):
            result = consolidate(
                client, guideline, candidates, output, 2, run_jobs, lambda s: None
            )
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][0]["source_batch_numbers"], [1, 2])
        self.assertEqual(result["groups"][1]["space"], candidates[2]["space"])
        self.assertEqual(read_json(output / "canonical_context.json")["batch_count"], 3)
        self.assertEqual(
            read_json(output / "canonical_context.json")["selection"][
                "reserved_output_tokens"
            ],
            100000,
        )

    def test_selection_cannot_invent_or_rewrite_clinical_definitions(self):
        lean = {k: state()[k] for k in LEAN_STATE["properties"]}
        lean["source_batch_numbers"] = [1]
        value = {
            "spaces": [copy.deepcopy(lean["space"])],
            "context_only_topics": [],
            "uncertainties": [],
        }
        self.assertEqual(selected_states(value, [lean]), [lean])
        value["spaces"][0]["prior_treatment_required"] = (
            "Invented treatment requirement"
        )
        with self.assertRaisesRegex(ValueError, "without rewriting"):
            selected_states(value, [lean])

    def test_candidate_allowlist_strips_arbitrary_internal_keys(self):
        candidate = {
            **state(),
            "candidate_id": "opaque-secret",
            "group_id": "opaque-group",
            "tracking": 19,
        }
        shown = clinical_candidate(candidate)
        for forbidden in (
            "opaque-secret",
            "opaque-group",
            "tracking",
            "candidate_id",
            "group_id",
        ):
            self.assertNotIn(forbidden, json.dumps(shown))
        self.assertEqual(shown["space"], candidate["space"])

    def test_selection_retry_explains_multiple_field_changes_without_record_ids(self):
        first = {k: state()[k] for k in LEAN_STATE["properties"]}
        first.update(name="Fictional localized population", candidate_id="opaque-first")
        second = copy.deepcopy(first)
        second.update(
            name="Fictional recurrent population", candidate_id="opaque-second"
        )
        second["space"]["cancer_burden_allowed"] = "Recurrent fictional cancer"
        value = {
            "spaces": [copy.deepcopy(s["space"]) for s in [first, second]],
            "context_only_topics": [],
            "uncertainties": [],
        }
        value["spaces"][0]["prior_treatment_required"] = "Invented treatment"
        value["spaces"][1]["sex_allowed"] = "Invented restriction"
        original = copy.deepcopy(value)
        with self.assertRaises(ValueError) as raised:
            selected_states(value, [first, second])
        feedback = str(raised.exception)
        self.assertIn("2 returned definitions", feedback)
        for text in (
            first["name"],
            second["name"],
            "prior_treatment_required",
            "sex_allowed",
            "Invented treatment",
            "Invented restriction",
        ):
            self.assertIn(text, feedback)
        self.assertNotIn("opaque-", feedback)
        self.assertNotIn("candidate_id", feedback)
        self.assertLessEqual(len(feedback), 1200)
        self.assertEqual(value, original)

    def test_selection_retry_guidance_is_bounded_for_long_clinical_fields(self):
        lean = {k: state()[k] for k in LEAN_STATE["properties"]}
        value = {
            "spaces": [copy.deepcopy(lean["space"])],
            "context_only_topics": [],
            "uncertainties": [],
        }
        value["spaces"][0]["prior_treatment_required"] = (
            "Unsupported requirement " * 200
        )
        with self.assertRaises(ValueError) as raised:
            selected_states(value, [lean])
        self.assertLessEqual(len(str(raised.exception)), 1200)

    def test_local_batch_can_contain_only_context(self):
        value = {
            "states": [],
            "context_only_topics": ["General background"],
            "uncertainties": [],
        }
        validate_catalog(value, load_guideline(self.root, "fictional").pages)


class PipelineTests(LibraryFixture, unittest.TestCase):
    def test_detail_repeats_fixed_population_and_candidate_alternatives_after_source(
        self,
    ):
        g = load_guideline(self.root, "fictional")
        candidate = {"candidate_id": "opaque-secret", **state()}
        candidate["treatment_options"] = [
            {
                "name": "Synthetic alternative A",
                "conditions": "After prior therapy in a different setting",
                "category": "Not specified",
                "evidence": [{"page_id": "p0002", "line_ids": [1]}],
            }
        ]
        group = {k: candidate[k] for k in LEAN_STATE["properties"]}
        group["source_batch_numbers"] = [1]
        original = copy.deepcopy(candidate)
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        _, _, messages = build_detail_call(client, g, group, [candidate])
        text = messages[-1]["content"]
        self.assertLess(text.index(TEXT), text.index("FINAL FIXED POPULATION"))
        self.assertLess(
            text.index("FINAL FIXED POPULATION"), text.index("Synthetic alternative A")
        )
        self.assertIn("NOT a claim that their populations", text)
        for forbidden in ("opaque-secret", "candidate_id", "source_batch_numbers"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(candidate, original)
        self.assertLessEqual(
            client.count_tokens(messages) + 100000 + 2048, client.config.context_window
        )

    def test_detail_candidate_page_overlap_is_not_a_clinical_identity_mapping(self):
        candidate = {"candidate_id": "first", **state()}
        other = copy.deepcopy(candidate)
        other["candidate_id"] = "second"
        other["space"]["prior_treatment_required"] = (
            "Different source-supported prior therapy"
        )
        unrelated = copy.deepcopy(candidate)
        unrelated["evidence"] = [{"page_id": "p9999", "line_ids": [1]}]
        result = candidate_menus(
            candidate, [candidate, copy.deepcopy(candidate), other, unrelated]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["space"], other["space"])

    def test_defining_branch_keeps_candidate_available_when_supporting_citations_differ(
        self,
    ):
        candidate = extraction()["candidates"][0]
        candidate["evidence"] = [{"page_id": "p0003", "line_ids": [1]}]
        result = candidate_menus(state(), [candidate])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["defining_branch"], candidate["defining_branch"])
        self.assertNotIn("candidate_id", result[0])

    def test_ownership_error_cannot_launch_citation_only_repair(self):
        g = load_guideline(self.root, "fictional")
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        with patch.object(
            client, "complete", side_effect=AssertionError("Must regenerate extraction")
        ):
            with self.assertRaisesRegex(ValueError, "Extraction ownership"):
                repair_response(
                    client,
                    g,
                    g.pages,
                    ["p0002"],
                    "extract-test",
                    extraction(),
                    "Extraction ownership: owner page must be states_extracted",
                    lambda value: None,
                )

    def test_later_frozen_branch_error_precedes_earlier_repairable_citation(self):
        g = load_guideline(self.root, "fictional")
        value = extraction()
        other = copy.deepcopy(value["candidates"][0])
        other["defining_branch"]["page_id"] = "p0001"
        value["candidates"].append(other)
        value["candidates"][0]["evidence"][0]["line_ids"] = [999]
        original = copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError, "Extraction ownership"):
            validate_extraction(value, g.pages, ["p0002"])
        self.assertEqual(value, original)

    def test_hidden_clinical_error_cannot_launch_citation_repair(self):
        g = load_guideline(self.root, "fictional")
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        value = extraction()
        value["candidates"][0]["evidence"][0]["line_ids"] = [999]
        value["candidates"][0]["space"]["cancer_burden_allowed"] = "A OR B AND C"
        original = copy.deepcopy(value)
        with patch.object(
            client, "complete", side_effect=AssertionError("Must regenerate extraction")
        ):
            with self.assertRaisesRegex(ValueError, "mixes AND and OR"):
                repair_response(
                    client,
                    g,
                    g.pages,
                    ["p0002"],
                    "extract-test",
                    value,
                    "p0002: nonexistent source line IDs [999]",
                    lambda value: None,
                )
        self.assertEqual(value, original)

    def test_inclusive_and_or_is_one_operator_without_changing_field_text(self):
        for text in (
            "A and/or B",
            "A AND / OR B",
            "A OR B and/or C",
            "A AND (B and/or C)",
            "(A AND B) OR C and/or D",
        ):
            with self.subTest(valid=text):
                value = state()
                value["space"]["biomarkers_required"] = text
                original = copy.deepcopy(value)
                validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)
        for text in ("A AND B and/or C", "A and/or B AND C", "(A AND B and/or C)"):
            with self.subTest(invalid=text):
                value = state()
                value["space"]["biomarkers_required"] = text
                original = copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError, "mixes AND and OR"):
                    validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_mixed_logic_retry_identifies_rejected_text_and_prose(self):
        value = state()
        text = "Suspected disease by symptoms or imaging; testing and staging pending"
        value["space"]["cancer_burden_allowed"] = text
        original = copy.deepcopy(value)
        with self.assertRaises(ValueError) as error:
            validate_decision_fields(value, "Fictional guideline")
        message = str(error.exception)
        self.assertIn(repr(text), message)
        self.assertIn("semicolons do not group logic", message)
        self.assertIn("corresponding considerations", message)
        self.assertEqual(value, original)

    def test_numeric_thresholds_are_atomic_without_changing_field_text(self):
        for text in (
            "(A OR B) AND score 2 or higher",
            "Score 2 or higher AND score 5 or lower",
            "Age 18 or older AND age 60 or younger",
            "A AND (score 2.5 or above OR B)",
            "A AND score 2 or more",
            "A AND score 2 or fewer",
            "A AND score 2 or greater",
            "A AND score 2 or less",
            "A AND score 2 or below.",
        ):
            with self.subTest(valid=text):
                value = state()
                value["space"]["cancer_burden_allowed"] = text
                original = copy.deepcopy(value)
                validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_numeric_thresholds_do_not_hide_population_alternatives(self):
        for text in (
            "A AND score 2 or higher OR B",
            "A AND score 2 OR higher-risk disease",
            "A AND score 2 OR lower grade disease",
            "A AND score 2 OR 3",
            "A AND (score 2 or higher OR B AND C)",
        ):
            with self.subTest(invalid=text):
                value = state()
                value["space"]["cancer_burden_allowed"] = text
                original = copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError, "mixes AND and OR"):
                    validate_decision_fields(value, "Fictional guideline")
                self.assertEqual(value, original)

    def test_extraction_retries_clinical_error_before_repairing_citations(self):
        g = load_guideline(self.root, "fictional")
        calls = []

        def respond(endpoint, body, **kwargs):
            prompt = body["messages"][1]["content"]
            self.assertFalse(prompt.startswith(REPAIR_TASK))
            if prompt.startswith(prompts.EXTRACT_TASK):
                value = extraction()
                calls.append("extract")
                if calls.count("extract") == 1:
                    value["candidates"][0]["evidence"][0]["line_ids"] = [999]
                    value["candidates"][0]["space"]["cancer_burden_allowed"] = (
                        "A OR B AND C"
                    )
                else:
                    self.assertIn("mixes AND and OR", body["messages"][-1]["content"])
            else:
                value = catalog() if prompt.startswith(CATALOG_TASK) else quoted_state()
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ]
            }

        output = self.base / "clinical-before-citations"
        with (
            patch.object(Client, "_http", side_effect=respond),
            patch("matchminer_ai.llm.structured.time.sleep"),
        ):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes", attempts=2),
                workers=1,
            )
        self.assertEqual(calls, ["extract", "extract"])
        self.assertEqual(audit_catalog(g, output)["accepted_requests"], 3)

    def test_detail_frozen_errors_precede_repairable_citations(self):
        g = load_guideline(self.root, "fictional")
        for criterion, expected in (
            ("Changed fictional population", "must exactly equal"),
            ("A OR B AND C", "mixes AND and OR"),
        ):
            with self.subTest(criterion=criterion):
                calls = []

                def respond(endpoint, body, **kwargs):
                    prompt = body["messages"][1]["content"]
                    self.assertFalse(prompt.startswith(REPAIR_TASK))
                    if prompt.startswith(prompts.EXTRACT_TASK):
                        value = extraction()
                    elif prompt.startswith(CATALOG_TASK):
                        value = catalog()
                    else:
                        value = quoted_state()
                        calls.append("detail")
                        if len(calls) == 1:
                            value["evidence"][0]["source_text"] = "Absent synthetic quotation"
                            value["space"]["cancer_burden_allowed"] = criterion
                        else:
                            self.assertIn(expected, body["messages"][-1]["content"])
                    return {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": json.dumps(value)},
                            }
                        ]
                    }

                output = self.base / ("detail-frozen-" + expected.replace(" ", "-"))
                with (
                    patch.object(Client, "_http", side_effect=respond),
                    patch("matchminer_ai.llm.structured.time.sleep"),
                ):
                    run_guideline(
                        g,
                        output,
                        LLMConfig(
                            model="synthetic", tokenizer_mode="bytes", attempts=2
                        ),
                        workers=1,
                    )
                self.assertEqual(calls, ["detail", "detail"])
                self.assertEqual(audit_catalog(g, output)["accepted_requests"], 3)

    def test_identical_definitions_union_lineage_without_collapsing_distinct_states(
        self,
    ):
        value = canonical()
        first = value["groups"][0]
        duplicate = copy.deepcopy(first)
        duplicate.update(
            candidate_ids=["c2"],
            name="Alternate model name",
            rationale="Second source rationale",
        )
        distinct = copy.deepcopy(first)
        distinct["candidate_ids"] = ["c3"]
        distinct["space"]["prior_treatment_required"] = "Prior fictional treatment"
        value["groups"].extend([duplicate, distinct])
        self.assertEqual(normalize_identical_groups(value), 1)
        self.assertEqual(len(value["groups"]), 2)
        self.assertEqual(first["candidate_ids"], ["extract-0001-c001", "c2"])
        self.assertIn("Second source rationale", first["rationale"])
        self.assertEqual(first["space"], state()["space"])
        validate_canonical(value, ["extract-0001-c001", "c2", "c3"])

    def fake_complete(
        self, client, job, messages, schema, validator, repair_handler=None
    ):
        rendered = json.dumps(messages)
        for forbidden in (
            "candidate_id",
            "group_id",
            "representative_id",
            "source_batch_numbers",
            "extract-0001-c001",
        ):
            self.assertNotIn(forbidden, rendered)
        response = (
            extraction()
            if schema == EXTRACTION
            else catalog()
            if schema == CATALOG
            else quoted_state()
        )
        validator(response)
        return response

    def test_pipeline_exports_compatible_table_and_evidence(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "output"
        with patch.object(
            Client, "complete", autospec=True, side_effect=self.fake_complete
        ):
            status = run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["paradigms"], 1)
        row = json.loads((output / "paradigms.jsonl").read_text())
        self.assertEqual(row["clinical_space_summary"], format_space(state()["space"]))
        self.assertTrue(row["review_required"])
        self.assertIn("#page=2", (output / "report.md").read_text())
        self.assertEqual(
            read_json(output / "coverage.json")["excluded_front_matter"], ["p0001"]
        )
        ledger = read_json(output / "extraction_ownership.json")
        self.assertEqual(ledger["page_owners"], {"p0002": "extract-0001"})
        self.assertEqual(ledger["branches"][0]["candidate_ids"], ["extract-0001-c001"])

    def test_changed_model_cannot_resume_same_output(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "output"
        with patch.object(
            Client, "complete", autospec=True, side_effect=self.fake_complete
        ):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                run_guideline(
                    g,
                    output,
                    LLMConfig(model="different", tokenizer_mode="bytes"),
                    workers=1,
                )

    def test_failure_is_explicit_without_final_catalog(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "output"
        with patch.object(
            Client, "complete", side_effect=EndpointError("synthetic failure")
        ):
            with self.assertRaises(RuntimeError):
                run_guideline(
                    g,
                    output,
                    LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                    workers=1,
                )
        self.assertEqual(read_json(output / "status.json")["status"], "failed")
        self.assertFalse((output / "paradigms.jsonl").exists())

    def test_prompt_requires_source_only_work_and_ambiguity_reporting(self):
        self.assertIn("ONLY the supplied sources", prompts.SYSTEM)
        self.assertIn("flowchart", prompts.SYSTEM)
        self.assertIn("EXACTLY unchanged", prompts.DETAIL_TASK)

    def test_population_rules_follow_source_and_payload_in_every_generation_stage(self):
        g = load_guideline(self.root, "fictional")
        g.metadata["title"] = "Synthetic Breast Cancer Guideline"
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        for task, schema in [
            (prompts.EXTRACT_TASK, EXTRACTION),
            (CATALOG_TASK, CATALOG),
            ("Select supplied clinical definitions", SELECTION),
            (prompts.DETAIL_TASK, DETAIL),
        ]:
            with self.subTest(schema=list(schema["properties"])):
                _, _, messages = pack_messages(
                    client,
                    g,
                    {"p0002"},
                    "fictional",
                    task,
                    {},
                    schema,
                    tail="FINAL CLINICAL PAYLOAD",
                )
                text = messages[-1]["content"]
                self.assertLess(text.index(TEXT), text.index("FINAL POPULATION RULES:"))
                self.assertLess(
                    text.index("FINAL CLINICAL PAYLOAD"),
                    text.index("FINAL POPULATION RULES:"),
                )
                for rule in (
                    "First-line, second-line and third-line-and-beyond",
                    "prior_treatment_required",
                    "FOR THAT ADVANCED SETTING",
                    "future progression",
                    "ER-negative AND PR-negative",
                    "(ER-positive OR PR-positive)",
                    "must NOT become",
                    "exactly one prior line",
                    "one OR two prior lines",
                    "Keep ordinal labels out of",
                    "cancer_burden_allowed. Do not infer",
                    "TRIALSPACE FIELD MEANINGS",
                    "visceral crisis OR endocrine refractory",
                    "These spaces\nmay overlap",
                    "Requirements in DIFFERENT fields apply together (AND)",
                    "'(A OR B) AND C'",
                    "restrictions in EVERY resulting alternative",
                    "name\n  is not part of the TrialSpace embedding text",
                ):
                    self.assertIn(rule, text)
                self.assertLessEqual(
                    client.count_tokens(messages) + 100000 + 2048,
                    client.config.context_window,
                )

    def test_field_descriptions_reach_real_selection_without_allowing_field_rewrites(
        self,
    ):
        from matchminer_ai.trials._guideline_canonical import build_selection_call

        g = load_guideline(self.root, "fictional")
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        states = catalog()["states"]
        _, _, messages = build_selection_call(client, g, states)
        text = messages[-1]["content"]
        self.assertIn("Visceral crisis is NOT prior treatment", text)
        self.assertIn("copied exactly", text)
        rewritten = copy.deepcopy(states[0]["space"])
        rewritten["cancer_burden_allowed"] = "Metastatic; visceral crisis"
        with self.assertRaisesRegex(ValueError, "without rewriting fields"):
            selected_states(
                {"spaces": [rewritten], "context_only_topics": [], "uncertainties": []},
                states,
            )

    def test_breast_wording_does_not_apply_to_other_guidelines_or_citation_repairs(
        self,
    ):
        g = load_guideline(self.root, "fictional")
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        _, _, messages = pack_messages(
            client, g, {"p0002"}, "fictional", "task", {}, EXTRACTION
        )
        self.assertNotIn("BREAST CANCER RECEPTOR WORDING", messages[-1]["content"])
        g.metadata["title"] = "Synthetic Breast Cancer Guideline"
        _, _, messages = pack_messages(
            client,
            g,
            {"p0002"},
            "fictional",
            "repair",
            {},
            EXTRACTION,
            population_guidance=False,
        )
        self.assertNotIn("FINAL POPULATION RULES", messages[-1]["content"])

    def test_new_prompt_version_requires_new_output_directory(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "old-prompt-output"
        config = LLMConfig(model="synthetic", tokenizer_mode="bytes")
        with patch.object(
            Client, "complete", autospec=True, side_effect=self.fake_complete
        ):
            with patch.object(prompts, "PROMPT_VERSION", "synthetic-earlier-prompt"):
                run_guideline(g, output, config, workers=1)
            with self.assertRaisesRegex(ValueError, "changed"):
                run_guideline(g, output, config, workers=1)

    def test_changed_prompt_text_cannot_reuse_cached_extraction_with_same_version_label(
        self,
    ):
        g = load_guideline(self.root, "fictional")
        output = self.base / "old-prompt-text"
        config = LLMConfig(model="synthetic", tokenizer_mode="bytes")
        with patch.object(
            Client, "complete", autospec=True, side_effect=self.fake_complete
        ):
            run_guideline(g, output, config, workers=1)
            with patch.object(
                prompts,
                "SPECIFICITY_RULES",
                prompts.SPECIFICITY_RULES + " New synthetic constraint.",
            ):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_guideline(g, output, config, workers=1)

    def test_changed_catalog_protocol_requires_new_output_directory(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "old-catalog-protocol"
        config = LLMConfig(model="synthetic", tokenizer_mode="bytes")
        with patch.object(
            Client, "complete", autospec=True, side_effect=self.fake_complete
        ):
            run_guideline(g, output, config, workers=1)
            with patch(
                "matchminer_ai.trials._guideline_pipeline.CATALOG_VERSION",
                "synthetic-new-protocol",
            ):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_guideline(g, output, config, workers=1)

    def test_context_packing_uses_whole_guideline_when_it_fits(self):
        g = load_guideline(self.root, "fictional")
        base = g.pages["p0002"]
        for number in range(3, 7):
            g.pages[f"p{number:04d}"] = replace(
                base, id=f"p{number:04d}", number=number, text=TEXT * 10
            )
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        selected, omitted, messages = pack_messages(
            client, g, {"p0002"}, "alpha", "task", {}, EXTRACTION
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(omitted, [])
        all_count = client.count_tokens(messages)
        client = Client(
            replace(client.config, context_window=100000 + 2048 + all_count - 1000),
            self.base / "cache",
        )
        selected, omitted, messages = pack_messages(
            client, g, {"p0002"}, "alpha", "task", {}, EXTRACTION
        )
        self.assertIn("p0002", [p.id for p in selected])
        self.assertTrue(omitted)
        self.assertLessEqual(
            client.count_tokens(messages) + 100000 + 2048, client.config.context_window
        )

    def test_focused_tail_is_counted_without_squeezing_output_reserve(self):
        g = load_guideline(self.root, "fictional")
        client = Client(
            LLMConfig(model="synthetic", tokenizer_mode="bytes"), self.base / "cache"
        )
        _, _, messages = pack_messages(
            client, g, {"p0002"}, "alpha", "task", {}, EXTRACTION
        )
        count = client.count_tokens(messages)
        client.config = replace(
            client.config, context_window=100000 + 2048 + count + 100
        )
        with self.assertRaises(ContextBudgetError):
            pack_messages(
                client, g, {"p0002"}, "alpha", "task", {}, EXTRACTION, tail="X" * 200
            )

    def test_offline_audit_checks_exports_and_rejects_altered_source_quotes(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "audited-output"

        def respond(endpoint, body, **kwargs):
            prompt = body["messages"][1]["content"]
            value = (
                extraction()
                if prompt.startswith(prompts.EXTRACT_TASK)
                else catalog()
                if prompt.startswith(CATALOG_TASK)
                else quoted_state()
            )
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ]
            }

        with patch.object(Client, "_http", side_effect=respond):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
        result = audit_catalog(g, output)
        self.assertEqual(result["verified_evidence_items"], 2)
        self.assertEqual(result["accepted_requests"], 3)
        original_ledger = read_json(output / "extraction_ownership.json")
        tampered_ledger = copy.deepcopy(original_ledger)
        tampered_ledger["branches"][0]["owner_job"] = "other-owner"
        atomic_json(output / "extraction_ownership.json", tampered_ledger)
        with self.assertRaisesRegex(ValueError, "ownership ledger differs"):
            audit_catalog(g, output)
        atomic_json(output / "extraction_ownership.json", original_ledger)
        # The prompt-only catalog upgrade must not make earlier content-schema runs unauditable.
        original_catalog = read_json(output / "canonical_groups.json")
        atomic_json(
            output / "canonical_groups.json",
            {**original_catalog, "version": "clinical-content-v1"},
        )
        self.assertEqual(audit_catalog(g, output)["verified_evidence_items"], 2)
        atomic_json(output / "canonical_groups.json", original_catalog)
        row = json.loads((output / "paradigms.jsonl").read_text())
        row["evidence"][0]["quote"] = "Altered fictional evidence"
        (output / "paradigms.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "source_text does not occur"):
            audit_catalog(g, output)

    def test_audit_derives_citation_lines_in_source_reading_order(self):
        root = make_library(
            self.base / "multiline", TEXT + "\nAdditional synthetic evidence."
        )
        g = load_guideline(root, "fictional")
        output = self.base / "sorted-evidence-output"

        def respond(endpoint, body, **kwargs):
            prompt = body["messages"][1]["content"]
            if prompt.startswith(prompts.EXTRACT_TASK):
                value = extraction()
            elif prompt.startswith(CATALOG_TASK):
                value = catalog()
            else:
                value = quoted_state()
                value["diagnostic_workup"][0]["evidence"][0]["source_text"] = (
                    TEXT + "\nAdditional synthetic evidence."
                )
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ]
            }

        with patch.object(Client, "_http", side_effect=respond):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
        row = json.loads((output / "paradigms.jsonl").read_text())
        self.assertEqual(row["diagnostic_workup"][0]["evidence"][0]["line_ids"], [1, 2])
        self.assertEqual(audit_catalog(g, output)["verified_evidence_items"], 2)

    def test_targeted_model_repair_preserves_clinical_records(self):
        g = load_guideline(self.root, "fictional")
        output = self.base / "repaired-output"
        calls = []

        def respond(endpoint, body, **kwargs):
            prompt = body["messages"][1]["content"]
            if prompt.startswith(REPAIR_TASK):
                calls.append("repair")
                value = {
                    "evidence_replacements": [
                        {
                            "candidate_index": 0,
                            "kind": "state",
                            "option_index": 0,
                            "evidence": [{"page_id": "p0002", "line_ids": [1]}],
                        }
                    ],
                    "coverage_replacements": [],
                    "unresolved": [],
                }
            elif prompt.startswith(prompts.EXTRACT_TASK):
                calls.append("extract")
                value = extraction()
                value["candidates"][0]["evidence"][0]["line_ids"] = [999]
            else:
                value = catalog() if prompt.startswith(CATALOG_TASK) else quoted_state()
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ]
            }

        with patch.object(Client, "_http", side_effect=respond):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
        self.assertEqual(calls, ["extract", "repair"])
        candidate = read_json(output / "candidates.json")[0]
        self.assertEqual(
            {k: v for k, v in candidate.items() if k != "candidate_id"},
            extraction()["candidates"][0],
        )
        accepted = [
            read_json(p) for p in (output / "checkpoints").glob("*/accepted.json")
        ]
        self.assertTrue(
            next(a for a in accepted if a["job"] == "extract-0001")["repair_applied"]
        )
        self.assertEqual(audit_catalog(g, output)["accepted_requests"], 4)
        attempts = len(list((output / "checkpoints").glob("*/attempt-*.json")))
        with patch.object(
            Client,
            "_http",
            side_effect=AssertionError("Repaired checkpoints must resume offline"),
        ):
            run_guideline(
                g,
                output,
                LLMConfig(model="synthetic", tokenizer_mode="bytes"),
                workers=1,
            )
        self.assertEqual(
            len(list((output / "checkpoints").glob("*/attempt-*.json"))), attempts
        )

    def test_repair_cannot_change_clinical_text_or_target_unknown_indexes(self):
        correction = {
            "evidence_replacements": [],
            "coverage_replacements": [],
            "unresolved": [],
        }
        correction["space"] = state()["space"]
        with self.assertRaises(ValueError):
            apply_repair(extraction(), correction, ["p0002"], is_extraction=True)
        correction.pop("space")
        correction["evidence_replacements"] = [
            {
                "candidate_index": 9,
                "kind": "state",
                "option_index": 0,
                "evidence": [{"page_id": "p0002", "line_ids": [1]}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            apply_repair(extraction(), correction, ["p0002"], is_extraction=True)

    def test_repair_targets_locate_all_blank_and_nonexistent_citations(self):
        pages = load_guideline(self.root, "fictional").pages
        pages["p0002"] = replace(pages["p0002"], text=TEXT + "\n\n")
        value = extraction()
        value["candidates"][0]["evidence"][0]["line_ids"] = [999]
        value["candidates"][0]["diagnostic_workup"][0]["evidence"][0]["line_ids"] = [2]
        targets = validation_targets(value, pages, ["p0002"], is_extraction=True)
        self.assertEqual(len(targets["evidence_errors"]), 2)
        self.assertEqual(targets["evidence_errors"][0]["nonexistent_line_ids"], [999])
        self.assertEqual(targets["evidence_errors"][1]["blank_line_ids"], [2])
        self.assertEqual(targets["evidence_errors"][1]["kind"], "diagnostic_workup")
        # The valid defining branch accounts for this primary page independently
        # of the two invalid supporting citations, which still need repair.
        self.assertEqual(targets["coverage_errors"], [])
        del value["candidates"][0]["defining_branch"]
        legacy_targets = validation_targets(value, pages, ["p0002"], is_extraction=True)
        self.assertEqual(legacy_targets["coverage_errors"][0]["page_id"], "p0002")


if __name__ == "__main__":
    unittest.main()
