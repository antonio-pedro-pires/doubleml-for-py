import pandas as pd
import pytest

from doubleml import DoubleMLMEDS
from doubleml.med.datasets import make_med_data
from doubleml.med.utils._meds_utils import generate_effects_summary

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.mark.ci
def test_effects_summary_stacks_the_summaries_of_all_effects(learner_linear):
    """effects_summary has one row per effect, holding the row of the summary of that effect."""
    data = make_med_data(n_obs=300, dim_x=5)
    meds_obj = DoubleMLMEDS(obj_dml_data=data, n_folds=2, n_folds_inner=2, **learner_linear)
    meds_obj.fit()

    effects_summary = meds_obj.effects_summary

    assert list(effects_summary.index) == ["ATE", "DIR_TREAT", "DIR_CONTROL", "INDIR_TREAT", "INDIR_CONTROL"]
    for effect_name, effect_framework in meds_obj.effects.items():
        pd.testing.assert_series_equal(effects_summary.loc[effect_name], effect_framework.summary.iloc[0], check_names=False)


@pytest.mark.ci
def test_generate_effects_summary_without_effects():
    """Without effects, the summary is an empty data frame. This case cannot be reached through DoubleMLMEDS."""
    pd.testing.assert_frame_equal(generate_effects_summary({}), pd.DataFrame(index=[]))
