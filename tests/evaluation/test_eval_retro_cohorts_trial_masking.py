"""Trial-identifier masking tests on small fabricated texts and tables only. No real patient data.

Protocol numbers, NCT IDs, drug codes and titles below are invented.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest

import matchminer_ai.evaluation.retro_cohorts.build_note_level_dataset as cli
from matchminer_ai.evaluation.retro_cohorts.notes import build_note_level_dataset, scan_residual_identifiers  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts.trial_masking import (TrialIdentifiers, event_trial_identifiers, mask_trial_identifiers,  # noqa: E402
                                          residual_flags, title_codes)

IDS = TrialIdentifiers(frozenset({17123}), frozenset({"1234567"}), ("ABC1234", "XYZ98765"))


@pytest.mark.parametrize("written", [
    "17-123", "17‐123", "17‑123", "17‒123", "17–123", "17—123", "17 - 123", "17 -123", "17- 123", "17 – 123",
    "DFCI 17-123", "DFHCC 17-123", "DFHCC #17-123", "protocol 17-123", "Protocol #: 17–123", "PROTOCOL 17 - 123",
    "#17-123", "# 17-123", "DFCI/DFHCC protocol 17-123", "DFCI17-123", "dfci protocol #17-123",
])
def test_own_protocol_masked_in_every_recognized_form(written):
    text, n_protocol, n_nct, n_code = mask_trial_identifiers(f"Discussed {written} today.", IDS)
    assert (text, n_protocol, n_nct, n_code) == ("Discussed [TRIAL PROTOCOL] today.", 1, 0, 0)


@pytest.mark.parametrize("written,expected", [
    ("DFCI 17123", "[TRIAL PROTOCOL]"), ("protocol #17123", "[TRIAL PROTOCOL]"), ("#17123", "[TRIAL PROTOCOL]"),
    ("17-123-01", "[TRIAL PROTOCOL]"), ("protocol 17-123/A", "[TRIAL PROTOCOL]"), ("DFCI 17-123-2", "[TRIAL PROTOCOL]"),
    ("DFCI/17-123", "DFCI/[TRIAL PROTOCOL]"), ("DFHCC-17-123", "DFHCC-[TRIAL PROTOCOL]"),
])
def test_own_protocol_masked_in_loose_spellings(written, expected):
    """2026-11-14 rerun: the event's own number without a dash after a prefix, with a suffix, or after a letter."""
    text, n_protocol, _, _ = mask_trial_identifiers(f"Discussed {written} today.", IDS)
    assert (text, n_protocol) == (f"Discussed {expected} today.", 1)


@pytest.mark.parametrize("written", ["DFCI 17124", "17-124-01", "DFCI/17-124", "#17124"])
def test_loose_spellings_of_other_protocols_not_masked(written):
    text = f"Discussed {written} today."
    assert mask_trial_identifiers(text, IDS) == (text, 0, 0, 0)


@pytest.mark.parametrize("written", [
    "17-124",            # a different protocol
    "18-123", "05-123",  # other NN-NNN numbers
    "117-123", "17-1234", "4.17-123", "1/17-123",  # part of a longer number
    "2017-12-31", "12-17-2023", "17-12-2023", "1/17/2023", "2017-123",  # dates
    "Platelets 150-400", "WBC 4.5-11.0", "ALT 17.0-123", "BP 117/123",  # lab ranges and vitals
    "617-123-4567",      # a phone number containing the digits
    "17123",             # no dash (the leakage regex does not recognize it; see the residual checks)
])
def test_look_alikes_not_masked(written):
    text = f"Seen: {written}."
    assert mask_trial_identifiers(text, IDS) == (text, 0, 0, 0)


@pytest.mark.parametrize("written", ["NCT01234567", "nct01234567", "NCT 01234567", "NCT1234567", "Nct 1234567"])
def test_own_nct_masked(written):
    assert mask_trial_identifiers(f"See {written}.", IDS) == ("See [NCT ID].", 0, 1, 0)


@pytest.mark.parametrize("written", ["NCT01234568", "NCT012345678", "NCT07654321", "01234567"])
def test_other_ncts_not_masked(written):
    text = f"See {written}."
    assert mask_trial_identifiers(text, IDS) == (text, 0, 0, 0)


def test_title_codes_and_code_matching():
    title = "A Phase 1/2 Study of ABC-1234 With or Without XYZ 98765 in Fictional Tumors During COVID 2019 (STAR-12)"
    assert title_codes(title) == ["ABC1234", "XYZ98765"]  # COVID excluded; STAR-12 has too few digits
    assert title_codes(None) == []
    assert title_codes("CD123-Directed ABC-1234 for CD137+ Disease") == ["ABC1234"]  # CD antigens are not drug codes
    for written in ["ABC-1234", "ABC1234", "abc 1234", "Abc–1234", "xyz98765", "XYZ-98765"]:
        assert mask_trial_identifiers(f"Started {written} on C1D1.", IDS) == ("Started [STUDY DRUG CODE] on C1D1.", 0, 0, 1)
    for written in ["ABC-12345", "XABC1234", "ABC-123", "DEF-1234", "ABC1234X", "XYZ987650"]:
        text = f"Started {written}."
        assert mask_trial_identifiers(text, IDS) == (text, 0, 0, 0)


def test_mixed_note_counts_and_only_own_identifiers():
    text = ("DFCI 17-123 (NCT01234567) screening; ABC-1234 dose 17-124 mg?; prior protocol 18-999; "
            "labs 10-200; abc1234 again; protocol 17–123.")
    masked, n_protocol, n_nct, n_code = mask_trial_identifiers(text, IDS)
    assert (n_protocol, n_nct, n_code) == (2, 1, 2)
    assert masked == ("[TRIAL PROTOCOL] ([NCT ID]) screening; [STUDY DRUG CODE] dose 17-124 mg?; prior protocol 18-999; "
                      "labs 10-200; [STUDY DRUG CODE] again; [TRIAL PROTOCOL].")
    assert mask_trial_identifiers(text, TrialIdentifiers()) == (text, 0, 0, 0)
    assert mask_trial_identifiers(None, IDS) == (None, 0, 0, 0)


def test_residual_flags_reuse_the_leakage_checks():
    assert residual_flags("Consented to 17-123, NCT01234567, ABC-1234.", IDS)["own_protocol"]
    clean = residual_flags(mask_trial_identifiers("Consented to 17-123, NCT01234567, ABC-1234, abc 1234.", IDS)[0], IDS)
    assert not any(clean.values())
    # The leakage regex is case-sensitive for codes; the mask (and own_code_any_case) is not.
    lower = residual_flags("started abc 1234", IDS)
    assert not lower["own_code"] and lower["own_code_any_case"]
    # Spellings the leakage regexes do not recognize are detected; three are masked since the 2026-11-14 rerun.
    for written, flag in (("DFCI 17123", "protocol_dashless_after_prefix"),
                          ("protocol 17-123-01", "protocol_followed_by_dash_or_slash"),
                          ("DFCI/17-123", "protocol_after_dot_dash_or_slash")):
        assert residual_flags(written, IDS)[flag]
        assert not residual_flags(mask_trial_identifiers(written, IDS)[0], IDS)[flag]
    # Bare "17 123" is too ambiguous to mask: reported only.
    assert residual_flags(mask_trial_identifiers("17 123", IDS)[0], IDS)["protocol_other_separator"]
    assert residual_flags("trial 01234567", IDS)["nct_digits_without_prefix"]
    assert residual_flags("ABC / 1234", IDS)["code_other_separator"]


def _write_reports(directory: Path):
    reports = pd.DataFrame(dict(
        dfci_mrn=[1, 1, 1, 2, 2],
        date=pd.to_datetime(["2019-12-01 09:00", "2020-02-01 09:00", "2020-06-01 09:00", "2020-01-01 09:00", "2020-01-02 00:00"]),
        note_type=["clinical", "imaging", "clinical", "clinical", "path"],
        RPT_ID=["a", "b", "c", "d", "e"],
        text=["Consented to DFCI 17-123 and protocol #05-678; NCT01234567 reviewed; ABC-1234 planned. "
              "Platelets 150-400, BP 120/80, prior trial 18-999.",
              "Follow-up imaging. Labs 10-200. Next visit 2020-03-01.",
              "Screening for protocol 18-999 (NCT00000999).",
              "Seen for 17-123 by another team; 21-321 discussed.",
              None]))
    directory.mkdir(parents=True, exist_ok=True)
    for kind, name in [("imaging", "all_imaging_reports"), ("clinical", "all_clinical_notes"), ("path", "all_path_reports")]:
        reports[reports.note_type.eq(kind)].to_parquet(directory / f"{name}.parquet", index=False, row_group_size=1)


def _events():
    # pseudo_mrn 1 has two protocols starting the same day; patient 1 has a later start (pseudo_mrn 2) on 18-999.
    return pd.DataFrame(dict(
        pseudo_mrn=[1, 1, 2, 3], dfci_mrn=[1, 1, 1, 2],
        protocol_number=["17-123", "05-678", "18-999", "21-321"],
        nct_id=["NCT01234567", "NCT07654321", "NCT00000999", "NCT02222222"],
        trial_start_dt=pd.to_datetime(["2020-03-20", "2020-03-20", "2020-08-01", "2020-05-01"]),
        note_anchor_dt=pd.to_datetime(["2020-03-01", "2020-03-01", "2020-07-15", "2020-05-01"])))


TITLES = {"NCT01234567": "A Phase 1 Study of ABC-1234 in Fictional Tumors", "NCT07654321": "A Study of DEF-5678",
          "NCT00000999": "Fictional Regimen in Fictional Disease", "NCT02222222": None}


def test_build_masks_each_event_own_trials_only(tmp_path):
    _write_reports(tmp_path / "derived")
    events = _events()
    out = tmp_path / "notes.parquet"
    stats = build_note_level_dataset(events, tmp_path / "derived", out, -14, anchor_col="note_anchor_dt",
                                     mask=True, titles=TITLES, batch_rows=1)
    got = pd.read_parquet(out)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert list(got.columns)[-3:] == ["n_masked_protocol", "n_masked_nct", "n_masked_code"]
    by = got.set_index(["pseudo_mrn", "RPT_ID"]).sort_index()
    assert by.index.tolist() == [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (2, "c"), (3, "d"), (3, "e")]
    # Event 1 masks both of its protocols, its NCT ID and its title code, and nothing of event 2's trial.
    assert by.loc[(1, "a"), "text"] == ("Consented to [TRIAL PROTOCOL] and [TRIAL PROTOCOL]; [NCT ID] reviewed; "
                                        "[STUDY DRUG CODE] planned. Platelets 150-400, BP 120/80, prior trial 18-999.")
    assert by.loc[(1, "a"), ["n_masked_protocol", "n_masked_nct", "n_masked_code"]].tolist() == [2, 1, 1]
    # The same note, kept for the patient's later start, masks only that start's protocol.
    assert by.loc[(2, "a"), "text"] == ("Consented to DFCI 17-123 and protocol #05-678; NCT01234567 reviewed; "
                                        "ABC-1234 planned. Platelets 150-400, BP 120/80, prior trial [TRIAL PROTOCOL].")
    assert by.loc[(2, "c"), "text"] == "Screening for [TRIAL PROTOCOL] ([NCT ID])."
    assert by.loc[(1, "b"), "text"] == "Follow-up imaging. Labs 10-200. Next visit 2020-03-01."
    # Patient 2's event masks its own protocol (21-321), not patient 1's (17-123).
    assert by.loc[(3, "d"), "text"] == "Seen for 17-123 by another team; [TRIAL PROTOCOL] discussed."
    assert pd.isna(by.loc[(3, "e"), "text"]) and by.loc[(3, "e"), "n_masked_protocol"] == 0
    m = stats["masking"]
    assert m["replacements"] == {"protocol": {"replacements": 5, "notes": 4, "index_events": 3},
                                 "nct": {"replacements": 2, "notes": 2, "index_events": 2},
                                 "code": {"replacements": 1, "notes": 1, "index_events": 1}}
    assert m["identifiers"]["index_events_with_several_protocols"] == 1
    assert m["identifiers"]["index_events_with_title_codes"] == 1  # event 1: ABC1234 and DEF5678 (never written)
    assert m["code_replacements_by_code"] == {"ABC1234": 1}
    assert sum(m["protocol_spellings_masked"].values()) == 5
    assert stats["rows"] == 7 and stats["rows_by_source"] == {"imaging": 2, "clinical": 4, "path": 1}
    assert stats["index_events_without_notes"] == 0
    # The worker pool writes the same rows in the same order.
    out2 = tmp_path / "notes2.parquet"
    stats2 = build_note_level_dataset(events, tmp_path / "derived", out2, -14, anchor_col="note_anchor_dt",
                                      mask=True, titles=TITLES, workers=2)
    pd.testing.assert_frame_equal(pd.read_parquet(out2), got)
    assert stats2 == stats
    # Residual check: nothing own survives the mask; without the mask the leaks are found.
    ids, _ = event_trial_identifiers(events, TITLES)
    clean = scan_residual_identifiers(out, ids)
    assert clean["rows_scanned"] == 7 and all(c["notes"] == 0 for c in clean["checks"].values())
    raw = tmp_path / "raw.parquet"
    build_note_level_dataset(events, tmp_path / "derived", raw, -14, anchor_col="note_anchor_dt")
    leaks = scan_residual_identifiers(raw, ids, workers=2)["checks"]
    assert leaks["own_protocol"] == {"notes": 4, "index_events": 3}
    assert leaks["own_nct"] == {"notes": 2, "index_events": 2} and leaks["own_code"] == {"notes": 1, "index_events": 1}


def test_masking_off_without_trial_columns(tmp_path):
    _write_reports(tmp_path / "derived")
    soc_like = _events()[["pseudo_mrn", "dfci_mrn", "trial_start_dt"]].drop_duplicates("pseudo_mrn")
    out = tmp_path / "soc.parquet"
    stats = build_note_level_dataset(soc_like, tmp_path / "derived", out, 5)
    got = pd.read_parquet(out)
    assert "n_masked_protocol" not in got and "masking" not in stats and not stats["mask"]
    assert got.set_index(["pseudo_mrn", "RPT_ID"]).loc[(3, "d"), "text"].startswith("Seen for 17-123")
    with pytest.raises(ValueError, match="protocol_number"):
        build_note_level_dataset(soc_like, tmp_path / "derived", out, 5, mask=True, titles={})
    with pytest.raises(ValueError, match="titles"):
        build_note_level_dataset(_events(), tmp_path / "derived", out, 5, mask=True)


def test_cli_masks_trial_cohort_by_default_and_writes_summary(tmp_path, capsys, monkeypatch):
    _write_reports(tmp_path / "derived")
    monkeypatch.setenv("MMAI_DATA_ROOT", str(tmp_path / "data"))
    phi = tmp_path / "data" / "phi"
    phi.mkdir(parents=True)
    pd.DataFrame([dict(nct_id=k, title=v) for k, v in TITLES.items()]).to_csv(tmp_path / "ctgov.csv", index=False)
    trial_csv = tmp_path / "trial.csv"
    _events().to_csv(trial_csv, index=False, date_format="%Y-%m-%d")
    common = ["--derived-data-dir", str(tmp_path / "derived"), "--ctgov-csv", str(tmp_path / "ctgov.csv"),
              "--workers", "1"]
    out = phi / "trial_minus14_masked.parquet"
    assert cli.main(["--anchor", "note_anchor", "--days-offset", "-14", "--denominator-csv", str(trial_csv),
                     "--output-parquet", str(out), *common]) == 0
    summary_path = phi / "trial_minus14_masked.summary.json"
    assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(summary_path.stat().st_mode) == 0o600
    summary = json.loads(summary_path.read_text())
    assert summary["parameters"]["mask"] and summary["residual_check"]["passed"]
    assert summary["counts"]["rows"] == 7 and summary["masking"]["replacements"]["protocol"]["replacements"] == 5
    assert "n_masked_code" in pd.read_parquet(out)
    # --no-mask, and SOC (no protocol_number / nct_id): masking off, no mask columns, still a summary.
    assert cli.main(["--days-offset", "-14", "--anchor", "note_anchor", "--denominator-csv", str(trial_csv),
                     "--output-parquet", str(phi / "unmasked.parquet"), "--no-mask", *common]) == 0
    assert "n_masked_code" not in pd.read_parquet(phi / "unmasked.parquet")
    soc_csv = tmp_path / "soc.csv"
    _events()[["pseudo_mrn", "dfci_mrn", "trial_start_dt"]].drop_duplicates("pseudo_mrn").to_csv(soc_csv, index=False)
    assert cli.main(["--days-offset", "5", "--denominator-csv", str(soc_csv),
                     "--output-parquet", str(phi / "soc.parquet"), *common]) == 0
    assert "Masking off: no protocol_number/nct_id columns" in capsys.readouterr().out
    soc_summary = json.loads((phi / "soc.summary.json").read_text())
    assert not soc_summary["parameters"]["mask"] and "residual_check" not in soc_summary
    # Note text is never written outside the data root's phi tree.
    with pytest.raises(SystemExit, match="outside a PHI directory"):
        cli.main(["--days-offset", "5", "--denominator-csv", str(soc_csv),
                  "--output-parquet", str(tmp_path / "elsewhere.parquet"), *common])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
