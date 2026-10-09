import pytest

from eegrep import synthetic
from eegrep.data.participants import analysis_table, assert_no_mmse, load_participants, model_covariates


def test_mmse_never_reaches_model_inputs(cfg, tmp_path):
    _, part = synthetic.make_recordings(n_per_class=2, duration_s=20.0)
    part.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    df = load_participants(tmp_path / "participants.tsv", cfg["dataset"]["group_to_label"])
    cov = model_covariates(df)
    assert "mmse" not in {c.lower() for c in cov.columns}
    assert "mmse" in analysis_table(df).columns     # analysis-only table keeps it


def test_assert_no_mmse_raises():
    with pytest.raises(AssertionError):
        assert_no_mmse(["age", "MMSE"])


def test_unknown_group_code_rejected(cfg, tmp_path):
    (tmp_path / "p.tsv").write_text("participant_id\tGender\tAge\tGroup\tMMSE\nsub-001\tF\t60\tX\t20\n")
    with pytest.raises(ValueError):
        load_participants(tmp_path / "p.tsv", cfg["dataset"]["group_to_label"])
