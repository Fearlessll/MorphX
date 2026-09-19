from configs.cohorts import get_cohort, parse_feature_families
from configs.features import feature_names


def test_hcc_channel_contract():
    spec = get_cohort("hcc")
    indices, names = parse_feature_families("all", spec)
    assert spec.total_channels == 168
    assert names == ["tissue", "texture", "nuclear"]
    assert indices == list(range(168))
    assert spec.tissue_labels[0] == "tumor"


def test_luad_channel_contract():
    spec = get_cohort("luad")
    indices, names = parse_feature_families("5+120", spec)
    assert spec.total_channels == 165
    assert names == ["tissue", "nuclear"]
    assert len(indices) == 125
    assert indices[:5] == list(range(5))
    assert indices[-1] == 164


def test_packaged_feature_names_follow_both_cohort_contracts():
    for cohort in ("hcc", "luad"):
        spec = get_cohort(cohort)
        names = feature_names(cohort)
        assert len(names) == spec.total_channels
        assert names[:spec.tissue_channels] == spec.tissue_labels
        assert names[-2:] == ("cell_num", "cell_area_ratio")
