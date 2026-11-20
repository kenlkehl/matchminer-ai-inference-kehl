"""Tests on small fabricated tables only. No real patient data."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from matchminer_ai.evaluation.retro_cohorts import enrollments as enr  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts import research_plans as rpl  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts import soc  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts import trials as tr  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts.io_utils import ctgov_date_bounds, protocol_number_to_int, write_csv_private  # noqa: E402
from matchminer_ai.evaluation.retro_cohorts.notes import build_note_level_dataset, count_prior_reports  # noqa: E402


def test_protocol_number_to_int_matches_v22_join_key():
    got = protocol_number_to_int(pd.Series(["17-123", "05-123", "TP1234", None, "12-345B"]))
    assert got.tolist()[:2] == [17123, 5123]
    assert got.isna().tolist() == [False, False, True, True, True]
    assert protocol_number_to_int(pd.Series(["5123", "17123"])).tolist() == [5123, 17123]


def test_ctgov_date_bounds_month_and_day_precision():
    first, last = ctgov_date_bounds(pd.Series(["2021-05", "2021-05-17", None, "bad"]))
    assert first[0] == pd.Timestamp("2021-05-01") and last[0] == pd.Timestamp("2021-05-31")
    assert first[1] == last[1] == pd.Timestamp("2021-05-17")
    assert first[2:].isna().all() and last[2:].isna().all()


def test_index_in_ctgov_window_widens_months_and_leaves_missing_unknown():
    idx = pd.Series(pd.to_datetime(["2021-05-31", "2021-06-01", "2021-06-01"]))
    got = tr.index_in_ctgov_window(idx, pd.Series(["2020-01", "2020-01", None]), pd.Series(["2021-05", "2021-05", "2022"]))
    assert bool(got[0])
    assert not bool(got[1])
    assert pd.isna(got[2])


def _oncore_csv(tmp_path):
    path = tmp_path / "oncore.csv"
    pd.DataFrame([
        dict(protocol_no="17-001", nct_id="NCT00000001", status="CLOSED TO ACCRUAL", phase="II", protocol_type="Treatment"),
        dict(protocol_no="17-002", nct_id="NCT00000001", status="OPEN TO ACCRUAL", phase="II", protocol_type="Treatment"),
        dict(protocol_no="05-003", nct_id="NCT00000003", status="IRB STUDY CLOSURE", phase="III", protocol_type="Treatment"),
        dict(protocol_no="18-004", nct_id=None, status="OPEN TO ACCRUAL", phase="I", protocol_type="Treatment"),
        dict(protocol_no="18-005", nct_id="NCT00000005", status="OPEN TO ACCRUAL", phase="NA", protocol_type="Banks / Registries"),
    ]).to_csv(path, index=False)
    return path


def _pe_csv(tmp_path):
    rows = []

    def add(mrn, seq, nbr, ttype, reg, on, start, open_dt="2015-01-01", close_dt=None):
        rows.append(dict(DFCI_MRN=str(mrn), PATIENT_STUDY_SEQ=str(seq), STUDY_NBR=nbr, TREATMENT_TYPE_CD=ttype,
                         REG_DT=reg, PROTOCOL_ON_DT=on, TREATMENT_START_DT=start,
                         STUDY_ACCRUAL_OPEN_DT=open_dt, STUDY_ACCRUAL_CLOSE_DT=close_dt))
    add(1, 1, "17001", "Tre", "2020-03-01 10:00:00", "2020-03-10 00:00:00", "2020-03-15 00:00:00")  # kept
    add(1, 2, "17002", "Tre", "2021-01-05 00:00:00", "2021-01-03 00:00:00", "2021-01-20 00:00:00")  # earliest = on-study
    add(2, 3, "5003", "Tre", "2025-12-31 09:00:00", "2026-01-05 00:00:00", "2026-01-10 00:00:00")   # last day kept
    add(3, 4, "5003", "Tre", "2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-01-03 00:00:00")   # after end
    add(4, 5, "17001", "Tre", "2015-12-31 00:00:00", "2016-01-04 00:00:00", "2016-01-05 00:00:00")  # before start
    add(5, 6, "17001", "Tre", "2019-06-01 00:00:00", "2019-06-02 00:00:00", None)                   # never started
    add(6, 7, "18004", "Tre", "2019-06-01 00:00:00", "2019-06-02 00:00:00", "2019-06-03 00:00:00")  # no NCT
    add(7, 8, "18005", "Bnk", "2019-06-01 00:00:00", "2019-06-02 00:00:00", "2019-06-03 00:00:00")  # not therapeutic
    add(8, 9, "17001", "Tre", "2022-02-01 00:00:00", "2022-02-02 00:00:00", "2022-02-03 00:00:00", close_dt="2021-12-31")
    add(9, 10, "0000", "Tre", "2024-02-01 00:00:00", "2024-02-02 00:00:00", "2024-02-03 00:00:00")  # masked active
    add(10, 11, "9006", "Tre", "2017-02-01 00:00:00", "2017-02-02 00:00:00", "2017-02-03 00:00:00")  # summary-only
    path = tmp_path / "pe.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_pull_enrollments_rules(tmp_path):
    oncore, counts = tr.load_oncore_treatment_protocols(_oncore_csv(tmp_path))
    assert counts["status_file_treatment_protocols_with_nct"] == 3
    assert oncore.set_index("protocol_number").first_protocol_for_nct.to_dict() == {
        "17-001": True, "17-002": False, "05-003": True}
    out, qualifying, attrition, diag = enr.pull_enrollments(
        _pe_csv(tmp_path), oncore, {"NCT00000001", "NCT00000003"}, "2016-01-01", "2025-12-31")
    assert sorted(out.dfci_mrn) == [1, 1, 2, 8]
    assert diag["masked_active_protocol_rows_in_window"] == 1
    assert diag["masked_active_protocol_rows_in_window_by_year"] == {"2024": 1}
    by_seq = out.set_index("patient_study_seq")
    assert by_seq.loc["2", "trial_start_dt"] == pd.Timestamp("2021-01-03")  # on-study before REG_DT
    assert by_seq.loc["1", "trial_start_dt"] == pd.Timestamp("2020-03-01")  # REG_DT floored to the day
    assert by_seq.loc["3", "trial_start_dt"] == pd.Timestamp("2025-12-31")
    assert bool(by_seq.loc["1", "index_in_dfci_accrual_window"])
    assert not bool(by_seq.loc["9", "index_in_dfci_accrual_window"])
    # pseudo_mrn numbered by (index date, mrn)
    assert out.sort_values("trial_start_dt").pseudo_mrn.tolist() == [1, 2, 3, 4]
    assert len(qualifying) == 6  # Tre + started + Treatment protocol with NCT + listing, any date
    assert [a["rows"] for a in attrition][:5] == [11, 10, 9, 9, 8]
    reg_only, _, _, _ = enr.pull_enrollments(_pe_csv(tmp_path), oncore, None, "2016-01-01", "2025-12-31", "reg_dt")
    assert reg_only.set_index("patient_study_seq").loc["2", "trial_start_dt"] == pd.Timestamp("2021-01-05")


def _tp_csv(tmp_path):
    rows = []

    def add(mrn, category, nbr, start, plan_id, name="Fictional research regimen"):
        rows.append(dict(DFCI_MRN=str(mrn), TREATMENT_PLAN_CATEGORY=category, RESEARCH_CHEMO_PLAN=name,
                         RESEARCH_CHEMO_PLAN_NBR=nbr, TPLAN_START_DT=start, FIRST_CYCLE_DT=start,
                         TPLAN_GOAL="PALLIATIVE", EPIC_TPLAN_ID=str(plan_id)))
    research = "ONCOLOGY RESEARCH CHEMO PLAN"
    add(1, research, "17001", "2020-03-20 00:00:00", 101)   # start; registered 2020-03-01 (same protocol)
    add(1, research, "17001", "2020-09-01 00:00:00", 102)   # re-plan of the same start
    add(9, research, "17002", "2024-02-20 00:00:00", 103)   # masked registration 2024-02-01 nearby
    add(11, research, "05003", "2023-05-01 00:00:00", 104)  # no registration at all
    add(4, research, "17001", "2015-12-01 00:00:00", 105)   # first start before the window
    add(4, research, "17001", "2016-02-01 00:00:00", 106)   # ...so this later plan is not a start
    add(12, research, "18005", "2019-01-01 00:00:00", 107)  # not a 'Treatment' protocol
    add(13, research, "18004", "2019-01-01 00:00:00", 108)  # Treatment protocol without an NCT ID
    add(14, research, None, "2019-01-01 00:00:00", 109)     # no protocol number
    add(15, "ONCOLOGY STANDARD CHEMO PLAN", None, "2019-01-01 00:00:00", 110)
    add(16, research, "17001", "2026-01-02 00:00:00", 111)  # after the window
    path = tmp_path / "tp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_pull_trial_starts_from_research_plans(tmp_path):
    oncore, _ = tr.load_oncore_treatment_protocols(_oncore_csv(tmp_path))
    assert set(oncore.oncore_protocol_type) == {"Treatment"}
    starts, first, attrition, diag = rpl.pull_trial_starts(
        _tp_csv(tmp_path), oncore, {"NCT00000001", "NCT00000003"}, "2016-01-01", "2025-12-31", _pe_csv(tmp_path))
    assert sorted(starts.dfci_mrn) == [1, 9, 11]
    by = starts.set_index("dfci_mrn")
    assert by.loc[1, "trial_start_dt"] == pd.Timestamp("2020-03-20") and by.loc[1, "n_research_plans_on_protocol"] == 2
    assert by.loc[1, "enrollment_table_match"] == "same_protocol" and by.loc[1, "registration_dt"] == pd.Timestamp("2020-03-01")
    assert by.loc[1, "days_registration_to_index"] == 19 and bool(by.loc[1, "index_in_dfci_accrual_window"])
    assert by.loc[9, "enrollment_table_match"] == "masked_registration_nearby"
    assert by.loc[11, "enrollment_table_match"] == "none" and pd.isna(by.loc[11, "registration_dt"])
    # Notes stop before the earliest known registration: same protocol, nearest masked one, else the index.
    assert by.loc[1, "note_anchor_dt"] == pd.Timestamp("2020-03-01")
    assert by.loc[9, "masked_registration_dt"] == pd.Timestamp("2024-02-01") == by.loc[9, "note_anchor_dt"]
    assert by.loc[11, "note_anchor_dt"] == pd.Timestamp("2023-05-01")
    assert by.loc[1, "index_date_rule"] == rpl.INDEX_DATE_RULE
    assert set(first.dfci_mrn) == {1, 4, 9, 11, 16}  # first starts of any date, Treatment + NCT + listing
    assert first.set_index("dfci_mrn").loc[4, "trial_start_dt"] == pd.Timestamp("2015-12-01")
    assert [a["rows"] for a in attrition][:5] == [11, 10, 9, 9, 7]
    assert diag["enrollment_table_match"] == {"same_protocol": 1, "masked_registration_nearby": 1, "none": 1}


def test_trial_table_and_window_flags(tmp_path):
    oncore, _ = tr.load_oncore_treatment_protocols(_oncore_csv(tmp_path))
    out, qualifying, _, _ = enr.pull_enrollments(_pe_csv(tmp_path), oncore, None, "2016-01-01", "2025-12-31")
    dates = pd.DataFrame(dict(nct_id=["NCT00000001"], ctgov_start_date=["2019-01"], ctgov_start_date_type=["ACTUAL"],
                              ctgov_completion_date=["2021-06-30"], ctgov_completion_date_type=["ESTIMATED"]))
    listings = pd.DataFrame(dict(nct_id=["NCT00000001"], title=["t"], brief_summary=["b"], eligibility_criteria=["e"],
                                 trial_text=["x"]))
    trials = enr.trial_table(out, qualifying, dates, listings, {"NCT00000003"}, set(), "2016-01-01", "2025-12-31")
    t1 = trials.set_index("nct_id").loc["NCT00000001"]
    assert t1.protocol_numbers == "17-001;17-002"
    # v22 window: treatment starts within the window (2016-01-05 from the pre-window registration counts)
    assert t1.first_dfci_treatment_start == pd.Timestamp("2016-01-05")
    assert t1.last_dfci_treatment_start == pd.Timestamp("2022-02-03")
    assert pd.isna(t1.dfci_accrual_close_dt)  # one protocol has no close date -> open-ended
    assert bool(trials.set_index("nct_id").loc["NCT00000003", "in_v23_training_sample"])
    flagged = enr.add_trial_window_flags(out, trials).set_index("patient_study_seq")
    assert bool(flagged.loc["1", "index_in_ctgov_window"]) and not bool(flagged.loc["9", "index_in_ctgov_window"])
    assert pd.isna(flagged.loc["3", "index_in_ctgov_window"])
    # NCT00000003's only in-window treatment start is 2026-01-10, after end_date
    assert pd.isna(flagged.loc["3", "index_in_v22_style_window"])


def test_summary_fills_missing_ncts(tmp_path):
    summary = tmp_path / "dfci.csv"
    pd.DataFrame([
        dict(protocol_no="17-001", nct_id="NCT00000099", status="CLOSED TO ACCRUAL", open_date=None, close_date=None),
        dict(protocol_no="18-004", nct_id="NCT00000004", status="OPEN TO ACCRUAL", open_date=None, close_date=None),
        dict(protocol_no="09-006", nct_id="nct00000006", status="IRB STUDY CLOSURE", open_date=None, close_date=None),
        dict(protocol_no="TP1234", nct_id="NCT00000007", status="NEW", open_date=None, close_date=None),
    ]).to_csv(summary, index=False)
    oncore, counts = tr.load_oncore_treatment_protocols(_oncore_csv(tmp_path), summary)
    got = oncore.set_index("protocol_number")
    assert got.loc["17-001", "nct_id"] == "NCT00000001" and got.loc["17-001", "nct_source"] == "oncore_status"
    assert got.loc["18-004", "nct_id"] == "NCT00000004" and got.loc["18-004", "nct_source"] == "dfci_trials_summary"
    assert got.loc["09-006", "nct_id"] == "NCT00000006" and pd.isna(got.loc["09-006", "first_protocol_for_nct"])
    assert "18-005" not in got.index  # in the status file but not 'Treatment'
    assert "TP1234" not in got.index
    out, _, _, diag = enr.pull_enrollments(_pe_csv(tmp_path), oncore, None, "2016-01-01", "2025-12-31")
    assert sorted(out.dfci_mrn) == [1, 1, 2, 6, 8, 10]
    assert diag["rows_by_nct_source"] == {"oncore_status": 4, "dfci_trials_summary": 2}


def test_dfci_holdout_rule(tmp_path):
    path = tmp_path / "dfci.csv"
    pd.DataFrame([
        dict(protocol_no="1", nct_id="NCT1", status="CLOSED TO ACCRUAL", open_date="2019-01", close_date="2024-12"),
        dict(protocol_no="2", nct_id="NCT2", status="CLOSED TO ACCRUAL", open_date="2019-01", close_date="2025-01"),
        dict(protocol_no="3", nct_id="NCT3", status="OPEN TO ACCRUAL", open_date="2010-01", close_date="2012-01"),
        dict(protocol_no="4", nct_id="NCT4", status="NEW", open_date=None, close_date=None),
        dict(protocol_no="5", nct_id="NCT5", status="NEW", open_date="2027-01", close_date=None),
    ]).to_csv(path, index=False)
    assert tr.dfci_holdout_nct_ids(path) == {"NCT2", "NCT3", "NCT4"}


def test_pull_soc_rules(tmp_path):
    rows = []

    def add(mrn, cat, std, research, goal, start, dx="C50.9"):
        rows.append(dict(DFCI_MRN=str(mrn), TREATMENT_PLAN_CATEGORY=cat, STD_CHEMO_PLAN=std, RESEARCH_CHEMO_PLAN=research,
                         RESEARCH_CHEMO_PLAN_NBR=None, TPLAN_START_DT=start, TPLAN_ICD_DX_CODES=dx, TPLAN_DX_NAME="x",
                         TPLAN_GOAL=goal, TPLAN_DC_REASON=None, FIRST_CYCLE_DT=None, EPIC_TPLAN_ID=f"p{len(rows)}",
                         TPLAN_SOURCE_NM="Epic"))
    std = "ONCOLOGY STANDARD CHEMO PLAN"
    add(1, std, "AC-T", None, "CURATIVE", "2016-01-01 00:00:00")     # kept, first day
    add(1, std, "AC-T", None, "PALLIATIVE", "2016-01-01 00:00:00")   # same day -> same pseudo_mrn
    add(2, std, "FOLFOX", None, "CONTROL", "2025-12-31 00:00:00")    # kept, last day
    add(2, std, "FOLFOX", None, "CONTROL", "2026-01-01 00:00:00")    # after end
    add(3, std, "X", None, None, "2020-01-01 00:00:00")              # no goal
    add(4, "ONCOLOGY RESEARCH CHEMO PLAN", None, "17-001 arm A", "CURATIVE", "2020-01-01 00:00:00")
    add(5, "ONCOLOGY RELATED OTHER PLAN", None, None, "CURATIVE", "2020-01-01 00:00:00")
    add(6, std, "Y", None, "CURATIVE", "2020-01-01 00:00:00")        # validation split
    add(7, std, "Z", None, "CURATIVE", "2020-01-01 00:00:00")        # no split row
    path = tmp_path / "tp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    split = pd.DataFrame(dict(dfci_mrn=[1, 2, 3, 4, 5, 6], split=["test"] * 5 + ["validation"]))
    out, attrition, all_splits = soc.pull_soc(path, split, "2016-01-01", "2025-12-31", "test")
    assert sorted(out.dfci_mrn) == [1, 1, 2]
    assert out.pseudo_mrn.nunique() == 2
    assert out.set_index("tplan_goal").is_palliative.to_dict() == {"CURATIVE": 0, "PALLIATIVE": 1, "CONTROL": 1}
    assert set(out.dx) == {"C50"} and set(out.is_trial) == {0}
    assert all_splits["validation"]["rows"] == 1
    everyone, _, _ = soc.pull_soc(path, split, "2016-01-01", "2025-12-31", None)
    assert sorted(everyone.dfci_mrn) == [1, 1, 2, 6]
    validation, _, _ = soc.pull_soc(path, split, "2016-01-01", "2025-12-31", "validation")
    assert list(validation.dfci_mrn) == [6] and list(validation.pseudo_mrn) == [1]


def _reports():
    return pd.DataFrame(dict(
        dfci_mrn=np.array([1, 1, 1, 2, 2, 3], dtype="int64"),
        date=pd.to_datetime(["2019-12-01 08:00", "2020-03-05 23:00", "2020-03-06 00:00",
                             "1950-01-01 00:00", "2030-01-01 00:00", "2020-01-01 00:00"]),
        note_type=["imaging", "clinical", "clinical", "path", "path", "imaging"],
        RPT_ID=["a", "b", "c", "d", "e", "f"], text=["t1", "t2", "t3", "t4", "t5", "t6"]))


def test_count_prior_reports_matches_naive_cutoff():
    reports = _reports()
    events = pd.DataFrame(dict(dfci_mrn=[1, 1, 2, 4], trial_start_dt=pd.to_datetime(["2020-03-01", "2019-01-01", "2020-01-01", "2020-01-01"])))
    got = count_prior_reports(events, reports[["dfci_mrn", "date"]], days_buffer=5)
    naive = [int(((reports.dfci_mrn == m) & (reports.date < d + pd.Timedelta(days=5))).sum())
             for m, d in zip(events.dfci_mrn, events.trial_start_dt)]
    assert got.tolist() == naive == [2, 0, 1, 0]


def test_build_note_level_dataset(tmp_path):
    reports = _reports()
    for name in ["all_imaging_reports", "all_clinical_notes", "all_path_reports"]:
        kind = {"all_imaging_reports": "imaging", "all_clinical_notes": "clinical", "all_path_reports": "path"}[name]
        reports[reports.note_type.eq(kind)].to_parquet(tmp_path / f"{name}.parquet", index=False)
    events = pd.DataFrame(dict(pseudo_mrn=[1, 2], dfci_mrn=[1, 2], trial_start_dt=pd.to_datetime(["2020-03-01", "2020-01-01"])))
    out = tmp_path / "notes.parquet"
    stats = build_note_level_dataset(events, tmp_path, out, days_offset=5, batch_rows=2)  # v22's cutoff
    got = pd.read_parquet(out).sort_values("RPT_ID")
    assert got.RPT_ID.tolist() == ["a", "b", "d"]
    assert stats["rows"] == 3 and (got.note_cutoff_dt == got.trial_start_dt + pd.Timedelta(days=5)).all()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    # Research-plan cohort: cut 7 days before the registration anchor, not after the plan start.
    anchored = events.assign(note_anchor_dt=pd.to_datetime(["2020-03-01", "2020-01-01"]))
    anchored = pd.concat([anchored, anchored.iloc[[0]].assign(note_anchor_dt=pd.Timestamp("2019-12-05"))])  # same event, earlier anchor
    build_note_level_dataset(anchored, tmp_path, out, days_offset=-1, batch_rows=2, anchor_col="note_anchor_dt")
    assert sorted(pd.read_parquet(out).RPT_ID) == ["a", "d"]  # 2019-12-01 < 2019-12-04; 1950 < 2019-12-31
    with pytest.raises(ValueError, match="explicit integer"):
        build_note_level_dataset(events, tmp_path, out, days_offset=None)


def test_sample_trials_reproduces_v22_draw():
    universe = [f"NCT{i:08d}" for i in range(50)][::-1]
    training = {"NCT00000001", "NCT00000002"}
    remaining = pd.DataFrame({"nct_id": [u for u in universe if u not in training], "x": 1})
    v22 = remaining.groupby("nct_id").first().reset_index()[["nct_id"]].sample(n=10, random_state=42).nct_id.tolist()
    assert tr.sample_trials(universe, training, 10, 42) == v22


def test_write_csv_private_sets_mode(tmp_path):
    path = write_csv_private(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
