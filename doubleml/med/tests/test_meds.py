import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml import DoubleMLMED
from doubleml.double_ml_framework import concat
from doubleml.med import DoubleMLMEDS
from doubleml.med.datasets import make_med_data

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")

# The DoubleMLMEDS object initializes multiple DoubleMLMED objects and computes the causal effects through them.
# The tests in this file compare a DoubleMLMEDS object with a collection of DoubleMLMED objects initialized
# with the same parameters. The tests pass if the collection of DoubleMLMED objects has the same qualities as
# the collection of objects initialized by DoubleMLMEDS and if the values computed through the collection of DoubleMLMED
# objects is identical to the values computed by DoubleMLMEDS.


@pytest.fixture(scope="module")
def meds_obj(dml_data, learner_linear, double_sample_splitting, ps_processor_config):
    return DoubleMLMEDS(
        obj_dml_data=dml_data,
        n_folds=5,
        n_rep=1,
        n_folds_inner=5,
        score="efficient-alt",
        normalize_ipw=False,
        draw_sample_splitting=True,
        double_sample_splitting=double_sample_splitting,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )


@pytest.fixture(scope="module")
def individual_med_objs(meds_obj, learner_linear, double_sample_splitting):
    smpls_inner = None if not meds_obj.double_sample_splitting else meds_obj.smpls_inner
    smpls = meds_obj._smpls

    individual_modeldict = {}
    for model_id, model in meds_obj.modeldict.items():
        ind_model = DoubleMLMED(
            obj_dml_data=meds_obj._dml_data,
            treatment_level=model.treatment_level,
            outcome=model.outcome,
            n_folds=meds_obj.n_folds,
            n_rep=meds_obj.n_rep,
            n_folds_inner=meds_obj.n_folds_inner,
            score=meds_obj.score,
            ps_processor_config=meds_obj.ps_processor_config,
            normalize_ipw=meds_obj.normalize_ipw,
            double_sample_splitting=meds_obj.double_sample_splitting,
            draw_sample_splitting=False,
            **learner_linear,
        )

        individual_modeldict[model_id] = ind_model
        ind_model.set_sample_splitting(smpls=smpls, smpls_inner=smpls_inner)
    return individual_modeldict


@pytest.mark.ci
def test_meds_modeldict(meds_obj):
    # Test that meds has the correct amount of models.
    n_models = len(meds_obj.models_ids)
    assert len(meds_obj.modeldict) == n_models


@pytest.mark.ci
def test_set_smpls(meds_obj, individual_med_objs):
    # Test that smpls and smpls_inner are correctly set for every model.
    reference_smpls = meds_obj.smpls
    reference_smpls_inner = meds_obj.smpls_inner

    [np.testing.assert_equal(reference_smpls, model.smpls) for _, model in individual_med_objs.items()]
    if meds_obj._double_sample_splitting:
        [
            (np.testing.assert_equal(reference_smpls_inner, model.smpls_inner) if model.outcome == "counterfactual" else None)
            for _, model in individual_med_objs.items()
        ]


@pytest.fixture(scope="module")
def fit_objs(meds_obj, individual_med_objs):
    np.random.seed(123)
    meds_obj.fit()

    np.random.seed(123)
    framework_list = [None] * len(individual_med_objs)
    for idx, model in enumerate(individual_med_objs.values()):
        framework_list[idx] = model.fit().framework
    individual_med_objs_framework = concat(framework_list)
    return meds_obj, individual_med_objs_framework


@pytest.mark.ci
def test_fit(fit_objs):
    # Test that the meds object models are fitted identically.
    meds_obj, individual_med_objs_framework = fit_objs
    [
        np.testing.assert_array_equal(
            meds_obj.framework.__getattribute__(elem), individual_med_objs_framework.__getattribute__(elem)
        )
        for elem in [
            "all_pvals",
            "all_ses",
            "all_t_stats",
            "all_thetas",
        ]
    ]


@pytest.mark.ci
def test_effects_binary_treats(fit_objs, individual_med_objs):
    meds_obj, _ = fit_objs
    effects_names = [
        "ATE",
        "DIR_TREAT",
        "DIR_CONTROL",
        "INDIR_TREAT",
        "INDIR_CONTROL",
    ]
    meds_obj.evaluate_effects()

    assert all([names == valid_names for names, valid_names in zip(meds_obj._effects, effects_names)])

    individual_effects = {
        "ATE": individual_med_objs["factual_1"].framework - individual_med_objs["factual_0"].framework,
        "DIR_TREAT": individual_med_objs["factual_1"].framework - individual_med_objs["counterfactual_0"].framework,
        "DIR_CONTROL": individual_med_objs["counterfactual_1"].framework - individual_med_objs["factual_0"].framework,
        "INDIR_TREAT": individual_med_objs["factual_1"].framework - individual_med_objs["counterfactual_1"].framework,
        "INDIR_CONTROL": individual_med_objs["counterfactual_0"].framework - individual_med_objs["factual_0"].framework,
    }

    for effects in effects_names:
        assert meds_obj._effects[effects].all_thetas == individual_effects[effects].all_thetas
        assert meds_obj._effects[effects].all_ses == individual_effects[effects].all_ses
        assert meds_obj._effects[effects].all_pvals == individual_effects[effects].all_pvals


@pytest.mark.ci
def test_bootstrap(fit_objs):
    meds_obj, individual_med_objs = fit_objs
    np.random.seed(123)
    meds_obj.bootstrap()
    np.random.seed(123)
    individual_med_objs.bootstrap()

    assert np.allclose(meds_obj.boot_t_stat, individual_med_objs.boot_t_stat)


@pytest.mark.ci
def test_effects_summary_reflects_latest_fit():
    """After fitting again, effects_summary is computed from the new fit and not taken from the previous one."""
    np.random.seed(3141)
    data = make_med_data(n_obs=300, dim_x=5)
    meds_obj = DoubleMLMEDS(
        data,
        ml_m=RandomForestClassifier(n_estimators=5),
        ml_g=RandomForestRegressor(n_estimators=5),
        ml_G=RandomForestRegressor(n_estimators=5),
        ml_M=RandomForestClassifier(n_estimators=5),
        ml_nested_g=RandomForestRegressor(n_estimators=5),
        n_folds=2,
        n_folds_inner=2,
    )

    meds_obj.fit()
    coef_factual_1_first = meds_obj.modeldict["factual_1"].coef.copy()
    meds_obj.effects_summary  # evaluates and caches the effects of the first fit

    meds_obj.fit()  # the random forests make the second fit differ from the first
    coef_factual_1 = meds_obj.modeldict["factual_1"].coef
    coef_factual_0 = meds_obj.modeldict["factual_0"].coef
    coef_counterfactual_0 = meds_obj.modeldict["counterfactual_0"].coef
    assert not np.allclose(coef_factual_1, coef_factual_1_first), "the second fit should differ from the first"

    effects_summary = meds_obj.effects_summary
    np.testing.assert_allclose(effects_summary.loc["ATE", "coef"], (coef_factual_1 - coef_factual_0)[0])
    np.testing.assert_allclose(effects_summary.loc["INDIR_CONTROL", "coef"], (coef_counterfactual_0 - coef_factual_0)[0])


@pytest.mark.ci
def test_meds_draw_sample_splitting_after_construction_without_splits(dml_data, learner_linear, double_sample_splitting):
    """Splits drawn after construction are used by all models, and the object can be fitted."""
    meds_obj = DoubleMLMEDS(
        obj_dml_data=dml_data,
        n_folds=2,
        n_folds_inner=2,
        double_sample_splitting=double_sample_splitting,
        draw_sample_splitting=False,
        **learner_linear,
    )

    assert meds_obj.draw_sample_splitting() is meds_obj
    meds_obj.fit()

    for model in meds_obj.modeldict.values():
        np.testing.assert_equal(model.smpls, meds_obj.smpls)
        if double_sample_splitting:
            np.testing.assert_equal(model.smpls_inner, meds_obj.smpls_inner)


@pytest.mark.ci
def test_meds_redraw_sample_splitting_updates_models(dml_data, learner_linear, double_sample_splitting):
    """Drawing new splits replaces the splits of all models and discards the results of the previous fit."""
    meds_obj = DoubleMLMEDS(
        obj_dml_data=dml_data,
        n_folds=2,
        n_folds_inner=2,
        double_sample_splitting=double_sample_splitting,
        **learner_linear,
    )
    meds_obj.fit()
    meds_obj.effects_summary  # evaluates and caches the effects of the first fit
    smpls_before = meds_obj.smpls

    np.random.seed(99)
    meds_obj.draw_sample_splitting()

    with pytest.raises(AssertionError):
        np.testing.assert_equal(meds_obj.smpls, smpls_before)  # the redraw gave different splits
    for model in meds_obj.modeldict.values():
        np.testing.assert_equal(model.smpls, meds_obj.smpls)
        if double_sample_splitting:
            np.testing.assert_equal(model.smpls_inner, meds_obj.smpls_inner)
    assert meds_obj.framework is None
    assert meds_obj.effects is None
