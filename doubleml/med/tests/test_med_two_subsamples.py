import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED, DoubleMLMEDData
from doubleml.med import DoubleMLMEDS
from doubleml.med.datasets import make_med_data

pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")

N_FOLDS = 3


@pytest.fixture(scope="module")
def small_data():
    np.random.seed(3141)
    return make_med_data(n_obs=300, dim_x=5)


def _med_obj(data, n_folds_inner, n_rep=1, outcome="counterfactual", **kwargs):
    np.random.seed(3141)
    return DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome=outcome,
        ml_m=LogisticRegression(),
        ml_g=LinearRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_rep=n_rep,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=True,
        **kwargs,
    )


@pytest.mark.ci
@pytest.mark.parametrize("n_rep", [1, 2])
def test_single_pair_draw_structure(small_data, n_rep):
    """n_folds_inner=1 draws one (A, B) pair per outer fold, disjoint and covering the outer training sample."""
    med_obj = _med_obj(small_data, n_folds_inner=1, n_rep=n_rep)

    assert med_obj.n_folds_inner == 1
    assert len(med_obj.smpls_inner) == n_rep
    for smpls_rep, inner_rep in zip(med_obj.smpls, med_obj.smpls_inner):
        assert len(inner_rep) == N_FOLDS
        for (train, _), inner_fold in zip(smpls_rep, inner_rep):
            assert len(inner_fold) == 1, f"Expected one inner (A, B) pair per outer fold, got {len(inner_fold)}."
            a_idx, b_idx = inner_fold[0]
            assert np.intersect1d(a_idx, b_idx).size == 0, "Subsamples A and B overlap."
            np.testing.assert_array_equal(np.sort(np.concatenate([a_idx, b_idx])), np.sort(train))


@pytest.mark.ci
def test_single_pair_draw_keeps_outer_folds(small_data):
    """The outer folds are the same as with two inner folds, and the pair is the first inner tuple."""
    single = _med_obj(small_data, n_folds_inner=1)
    two = _med_obj(small_data, n_folds_inner=2)

    np.testing.assert_equal(single.smpls, two.smpls)
    for inner_single, inner_two in zip(single.smpls_inner[0], two.smpls_inner[0]):
        np.testing.assert_equal(inner_single[0], inner_two[0])


@pytest.mark.ci
def test_zero_inner_folds_raises(small_data):
    with pytest.raises(ValueError, match="n_folds_inner"):
        _med_obj(small_data, n_folds_inner=0)


@pytest.mark.ci
def test_set_sample_splitting_accepts_single_pair(small_data):
    drawn = _med_obj(small_data, n_folds_inner=1)
    ext = _med_obj(small_data, n_folds_inner=5, draw_sample_splitting=False)

    ext.set_sample_splitting(smpls=drawn.smpls, smpls_inner=drawn.smpls_inner)

    assert ext.n_folds_inner == 1
    np.testing.assert_equal(ext.smpls_inner, drawn.smpls_inner)


def _single_pair_with(drawn, replace):
    inner = [[list(fold) for fold in rep] for rep in drawn.smpls_inner]
    inner[0][0][0] = replace(drawn.smpls[0][0][0], inner[0][0][0])
    return inner


@pytest.mark.ci
@pytest.mark.parametrize(
    "replace, match",
    [
        (lambda train, pair: (pair[0], pair[1][:-1]), "must together cover the outer training fold"),
        (lambda train, pair: (train[:-1], train[-2:]), "Intersection of train and test indices is not empty"),
    ],
    ids=["pair_not_covering_train", "overlapping_pair"],
)
def test_set_sample_splitting_rejects_invalid_pair(small_data, replace, match):
    drawn = _med_obj(small_data, n_folds_inner=1)
    ext = _med_obj(small_data, n_folds_inner=1, draw_sample_splitting=False)

    with pytest.raises(ValueError, match=match):
        ext.set_sample_splitting(smpls=drawn.smpls, smpls_inner=_single_pair_with(drawn, replace))


class _SpyRegressor(LinearRegression):
    """Linear regression that logs the covariate ids it is fitted and evaluated on, keyed by ``name``."""

    LOG = {}

    def __init__(self, name="spy"):
        super().__init__()
        self.name = name

    def fit(self, X, y, sample_weight=None):
        _SpyRegressor.LOG.setdefault(self.name, {"fit": [], "predict": []})["fit"].append((X[:, 0].copy(), np.copy(y)))
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        _SpyRegressor.LOG[self.name]["predict"].append(X[:, 0].copy())
        return super().predict(X)


@pytest.fixture(scope="module")
def spied_fit(small_data):
    _SpyRegressor.LOG.clear()
    drawn = _med_obj(small_data, n_folds_inner=1)
    np.random.seed(3141)
    med_obj = DoubleMLMED(
        obj_dml_data=small_data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_g=LinearRegression(),
        ml_G=_SpyRegressor(name="ml_G"),
        ml_M=LogisticRegression(),
        ml_nested_g=_SpyRegressor(name="ml_nested_g"),
        n_folds=N_FOLDS,
        n_folds_inner=1,
        double_sample_splitting=True,
        draw_sample_splitting=False,
    )
    med_obj.set_sample_splitting(smpls=drawn.smpls, smpls_inner=drawn.smpls_inner)
    med_obj.fit()
    return drawn, med_obj


def _ids(small_data, idx):
    return set(np.round(small_data.x[idx, 0], 12))


def _logged_ids(values):
    return set(np.round(values, 12))


@pytest.mark.ci
def test_single_pair_models_use_subsamples(small_data, spied_fit):
    """ml_G learns on A (treated), predicts B and the test fold; ml_nested_g learns on B (untreated)."""
    drawn, _ = spied_fit
    treated = small_data.d == 1
    log_G, log_nested = _SpyRegressor.LOG["ml_G"], _SpyRegressor.LOG["ml_nested_g"]

    assert len(log_G["fit"]) == N_FOLDS, "Expected one ml_G model per outer fold."
    assert len(log_nested["fit"]) == N_FOLDS
    for k, ((_, test), inner_fold) in enumerate(zip(drawn.smpls[0], drawn.smpls_inner[0])):
        a_idx, b_idx = inner_fold[0]
        assert _logged_ids(log_G["fit"][k][0]) == _ids(small_data, a_idx[treated[a_idx]])
        assert {frozenset(_logged_ids(v)) for v in log_G["predict"][2 * k : 2 * k + 2]} == {
            frozenset(_ids(small_data, b_idx)),
            frozenset(_ids(small_data, test)),
        }
        assert _logged_ids(log_nested["fit"][k][0]) == _ids(small_data, b_idx[~treated[b_idx]])
        assert _logged_ids(log_nested["predict"][k]) == _ids(small_data, test)


@pytest.mark.ci
def test_single_pair_nested_targets_are_ml_g_predictions_on_b(small_data, spied_fit):
    drawn, _ = spied_fit
    treated = small_data.d == 1
    xm = np.column_stack((small_data.x, small_data.m))
    for k, inner_fold in enumerate(drawn.smpls_inner[0]):
        a_idx, b_idx = inner_fold[0]
        a_treated, b_untreated = a_idx[treated[a_idx]], b_idx[~treated[b_idx]]
        expected = LinearRegression().fit(xm[a_treated], small_data.y[a_treated]).predict(xm[b_untreated])
        np.testing.assert_allclose(_SpyRegressor.LOG["ml_nested_g"]["fit"][k][1], expected)


def _fit_with_pair(data, drawn, pair_builder):
    train = drawn.smpls[0][0][0]
    treated = data.d == 1
    inner = [[list(fold) for fold in rep] for rep in drawn.smpls_inner]
    inner[0][0][0] = pair_builder(train, treated[train])
    ext = _med_obj(data, n_folds_inner=1, draw_sample_splitting=False)
    ext.set_sample_splitting(smpls=drawn.smpls, smpls_inner=inner)
    ext.fit()


@pytest.mark.ci
def test_single_pair_requires_treated_units_in_a(small_data):
    drawn = _med_obj(small_data, n_folds_inner=1)
    with pytest.raises(
        ValueError,
        match=(
            r"ml_G requires at least one observation with D=1 in subsample A of outer fold 0"
            r".*Redraw the sample splitting, use fewer folds or pass a stratified smpls_inner to set_sample_splitting\."
        ),
    ):
        _fit_with_pair(small_data, drawn, lambda train, is_treated: (train[~is_treated], train[is_treated]))


@pytest.mark.ci
def test_single_pair_requires_untreated_units_in_b(small_data):
    drawn = _med_obj(small_data, n_folds_inner=1)

    def treated_only_b(train, is_treated):
        b_idx = train[is_treated][:5]
        return np.setdiff1d(train, b_idx), b_idx

    with pytest.raises(
        ValueError,
        match=(
            r"ml_nested_g requires at least one observation with D!=1 in subsample B of outer fold 0"
            r".*Redraw the sample splitting, use fewer folds or pass a stratified smpls_inner to set_sample_splitting\."
        ),
    ):
        _fit_with_pair(small_data, drawn, treated_only_b)


def _meds_obj(data, n_folds_inner=1, **kwargs):
    np.random.seed(3141)
    return DoubleMLMEDS(
        obj_dml_data=data,
        ml_m=LogisticRegression(),
        ml_g=LinearRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=True,
        **kwargs,
    )


@pytest.mark.ci
def test_meds_single_pair_draw_and_forwarding(small_data):
    meds_obj = _meds_obj(small_data)

    assert meds_obj.n_folds_inner == 1
    for inner_fold, (train, _) in zip(meds_obj.smpls_inner[0], meds_obj.smpls[0]):
        assert len(inner_fold) == 1
        a_idx, b_idx = inner_fold[0]
        np.testing.assert_array_equal(np.sort(np.concatenate([a_idx, b_idx])), np.sort(train))
    for model in meds_obj.modeldict.values():
        assert model.n_folds_inner == 1
        np.testing.assert_equal(model.smpls_inner, meds_obj.smpls_inner)


@pytest.mark.ci
def test_meds_single_pair_fit_and_redraw(small_data):
    meds_obj = _meds_obj(small_data)
    meds_obj.fit()
    assert np.all(np.isfinite(meds_obj.coef))

    meds_obj.draw_sample_splitting()
    assert meds_obj.n_folds_inner == 1
    assert all(len(inner_fold) == 1 for inner_fold in meds_obj.smpls_inner[0])


@pytest.mark.ci
@pytest.mark.parametrize("n_folds", [2, 3])
def test_single_pair_nested_predictions_do_not_depend_on_own_fold_outcomes(n_folds):
    """Cross-fitting: changing the outcomes of the held-out fold leaves its nested predictions unchanged."""

    def build(frame, x_cols):
        data = DoubleMLMEDData(frame, y_col="y", d_cols="d", m_cols="m", x_cols=x_cols)
        np.random.seed(7)
        return DoubleMLMED(
            obj_dml_data=data,
            treatment_level=1,
            outcome="counterfactual",
            ml_m=LogisticRegression(),
            ml_G=LinearRegression(),
            ml_M=LogisticRegression(),
            ml_nested_g=LinearRegression(),
            n_folds=n_folds,
            n_folds_inner=1,
        )

    np.random.seed(3141)
    frame = make_med_data(n_obs=400, dim_x=5).data
    x_cols = [column for column in frame.columns if column.startswith("X")]
    reference = build(frame, x_cols)
    held_out = reference.smpls[0][0][1]
    reference.fit()

    shifted_frame = frame.copy()
    treated_held_out = held_out[frame["d"].to_numpy()[held_out] == 1]
    shifted_frame.loc[shifted_frame.index[treated_held_out], "y"] += 25.0
    shifted = build(shifted_frame, x_cols)
    shifted.set_sample_splitting(smpls=reference.smpls, smpls_inner=reference.smpls_inner)
    shifted.fit()

    np.testing.assert_allclose(
        shifted.predictions["ml_nested_g"][held_out, 0, 0],
        reference.predictions["ml_nested_g"][held_out, 0, 0],
        err_msg="the nested predictions of a fold depend on the outcomes of that fold",
    )


@pytest.mark.ci
def test_single_pair_stored_predictions_and_models(small_data):
    med_obj = _med_obj(small_data, n_folds_inner=1)
    med_obj.fit(store_models=True)

    for k, inner_fold in enumerate(med_obj.smpls_inner[0]):
        stored = med_obj.predictions[f"ml_G_inner_{k}"][:, 0, 0]
        b_idx = inner_fold[0][1]
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(stored)), np.sort(b_idx))
        assert len(med_obj.models[f"ml_G_inner_{k}"][small_data.d_cols[0]][0]) == 1, "Expected one ml_G model per outer fold."


@pytest.mark.ci
def test_single_pair_external_predictions_round_trip(small_data):
    med_obj = _med_obj(small_data, n_folds_inner=1)
    med_obj.fit()
    external = {"d": {name: med_obj.predictions[name][:, :, 0] for name in med_obj.params_names if name != "ml_g"}}

    med_ext = _med_obj(small_data, n_folds_inner=1, draw_sample_splitting=False)
    med_ext.set_sample_splitting(smpls=med_obj.smpls, smpls_inner=med_obj.smpls_inner)
    med_ext.fit(external_predictions=external)

    np.testing.assert_allclose(med_ext.coef, med_obj.coef)
    np.testing.assert_allclose(med_ext.se, med_obj.se)
