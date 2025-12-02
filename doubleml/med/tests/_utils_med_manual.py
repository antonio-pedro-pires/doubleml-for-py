import numpy as np
from sklearn.base import clone, is_classifier

from doubleml.tests._utils import fit_predict, fit_predict_proba
from doubleml.tests._utils_boot import boot_manual, draw_weights
from doubleml.utils._checks import _check_is_propensity


def fit_med_manual(
    y,
    x,
    d,
    m,
    learner_yx,
    learner_px,
    learner_ymx=None,
    learner_pmx=None,
    learner_nested=None,
    target="potential",
    treatment_level=1,
    mediation_level=1,
    all_smpls=None,
    n_rep=1,
    trimming_threshold=1e-2,
):
    if target not in ["potential", "counterfactual"]:
        raise ValueError(f"Invalid target: {target}")

    n_obs = len(y)
    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    all_preds = {
        "yx_hat": [],
        "px_hat": [],
        "ymx_hat": [],
        "pmx_hat": [],
        "nested_hat": [],
    }

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        preds = fit_nuisance_med_manual(
            y,
            x,
            d,
            m,
            learner_yx,
            learner_px,
            learner_ymx,
            learner_pmx,
            learner_nested,
            smpls,
            target,
            treatment_level,
            mediation_level,
            trimming_threshold,
        )

        for key, val in preds.items():
            if key in all_preds:
                all_preds[key].append(val)

        thetas[i_rep], ses[i_rep] = med_dml2_manual(
            y,
            d,
            m,
            preds,
            smpls,
            target,
            treatment_level,
            mediation_level,
            trimming_threshold,
        )

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {
        "theta": theta,
        "se": se,
        "thetas": thetas,
        "ses": ses,
    }
    res.update({f"all_{k}": v for k, v in all_preds.items() if v})

    return res


def fit_nuisance_med_manual(
    y,
    x,
    d,
    m,
    learner_yx,
    learner_px,
    learner_ymx,
    learner_pmx,
    learner_nested,
    smpls,
    target,
    treatment_level,
    mediation_level,
    trimming_threshold,
    yx_params=None,
    ymx_params=None,
    px_params=None,
    pmx_params=None,
    nested_params=None,
):
    if target == "potential":
        return _fit_nuisance_potential_manual(
            y,
            x,
            d,
            m,
            learner_yx,
            learner_px,
            smpls,
            treatment_level,
            trimming_threshold,
            yx_params,
            px_params,
        )
    else:
        return _fit_nuisance_counterfactual_manual(
            y,
            x,
            d,
            m,
            learner_yx,
            learner_px,
            learner_ymx,
            learner_pmx,
            learner_nested,
            smpls,
            treatment_level,
            mediation_level,
            trimming_threshold,
            yx_params,
            ymx_params,
            px_params,
            pmx_params,
            nested_params,
        )


def _fit_nuisance_potential_manual(
    y,
    x,
    d,
    m,
    learner_yx,
    learner_px,
    smpls,
    treatment_level,
    trimming_threshold,
    yx_params,
    px_params,
):
    treated = d == treatment_level
    ml_yx = clone(learner_yx)
    ml_px = clone(learner_px)

    train_cond_d1 = np.where(treated == 1)[0]
    if is_classifier(learner_yx):
        yx_hat_list = fit_predict_proba(y, x, ml_yx, yx_params, smpls, train_cond=train_cond_d1)
    else:
        yx_hat_list = fit_predict(y, x, ml_yx, yx_params, smpls, train_cond=train_cond_d1)

    px_hat_list = fit_predict_proba(treated, x, ml_px, px_params, smpls, trimming_threshold=trimming_threshold)

    return {"yx_hat": yx_hat_list, "px_hat": px_hat_list}


def _fit_nuisance_counterfactual_manual(
    y,
    x,
    d,
    m,
    learner_yx,
    learner_px,
    learner_ymx,
    learner_pmx,
    learner_nested,
    smpls,
    treatment_level,
    mediation_level,
    trimming_threshold,
    yx_params,
    ymx_params,
    px_params,
    pmx_params,
    nested_params,
):
    treated = d == treatment_level
    mediated = m == mediation_level

    ml_yx = clone(learner_yx)
    ml_px = clone(learner_px)
    ml_ymx = clone(learner_ymx)
    ml_pmx = clone(learner_pmx)
    xm = np.column_stack((x, m))

    px_hat_list = fit_predict_proba(treated, x, ml_px, px_params, smpls, trimming_threshold=trimming_threshold)

    pmx_hat_list = fit_predict_proba(treated, xm, ml_pmx, pmx_params, smpls, trimming_threshold=trimming_threshold)

    train_cond_d1_m1 = np.where((treated == 1) & (mediated == 1))[0]

    if is_classifier(ml_yx):
        yx_hat_list = fit_predict_proba(y, x, ml_yx, yx_params, smpls, train_cond=train_cond_d1_m1)
    else:
        yx_hat_list = fit_predict(y, x, ml_yx, yx_params, smpls, train_cond=train_cond_d1_m1)

    train_cond_d1 = np.where(treated == 1)[0]

    if is_classifier(ml_ymx):
        ymx_hat_list = fit_predict_proba(y, x, ml_ymx, ymx_params, smpls, train_cond=train_cond_d1)
    else:
        ymx_hat_list = fit_predict(y, x, ml_ymx, ymx_params, smpls, train_cond=train_cond_d1)

    return {
        "yx_hat": yx_hat_list,
        "px_hat": px_hat_list,
        "pmx_hat": pmx_hat_list,
        "ymx_hat": ymx_hat_list,
        # "nested_hat": nested_hat_list,
    }


def med_dml2_manual(
    y,
    d,
    m,
    preds,
    smpls,
    target,
    treatment_level,
    mediation_level,
    trimming_threshold,
):
    residuals = compute_residuals_manual(
        y,
        d,
        m,
        preds,
        smpls,
        target,
        treatment_level,
        mediation_level,
        trimming_threshold,
    )

    theta_hat = med_orth_manual(residuals, d, m, target, treatment_level, mediation_level)
    se = np.sqrt(var_med_manual(theta_hat, residuals, d, m, target, treatment_level, mediation_level))

    return theta_hat, se


def compute_residuals_manual(
    y,
    d,
    m,
    preds,
    smpls,
    target,
    treatment_level,
    mediation_level,
    trimming_threshold,
):
    if target == "potential":
        return _compute_residuals_potential_manual(y, d, preds, smpls, treatment_level)
    else:
        return _compute_residuals_counterfactual_manual(y, d, preds, smpls, treatment_level)


def _compute_residuals_potential_manual(y, d, preds, smpls, treatment_level):
    yx_hat = preds["yx_hat"]
    px_hat = preds["px_hat"]

    n_obs = len(y)
    u_hat = np.full(n_obs, np.nan)
    yx_hat_vec = np.full(n_obs, np.nan)
    px_hat_vec = np.full(n_obs, np.nan)

    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - yx_hat[idx]
        yx_hat_vec[test_index] = yx_hat[idx]
        px_hat_vec[test_index] = px_hat[idx]
    _check_is_propensity(px_hat_vec, "px_hat", "ml_px", smpls, eps=1e-12)

    return {
        "u_hat": u_hat,
        "yx_hat": yx_hat_vec,
        "px_hat": px_hat_vec,
    }


def _compute_residuals_counterfactual_manual(y, d, preds, smpls, treatment_level):
    ymx_hat = preds["ymx_hat"]
    nested_hat = preds["nested_hat"]
    px_hat = preds["px_hat"]
    pmx_hat = preds["pmx_hat"]

    n_obs = len(y)
    u_hat = np.full(n_obs, np.nan)
    w_hat = np.full(n_obs, np.nan)
    ymx_hat_vec = np.full(n_obs, np.nan)
    nested_hat_vec = np.full(n_obs, np.nan)
    px_hat_vec = np.full(n_obs, np.nan)
    pmx_hat_vec = np.full(n_obs, np.nan)

    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - ymx_hat[idx]
        w_hat[test_index] = ymx_hat[idx] - nested_hat[idx]

        ymx_hat_vec[test_index] = ymx_hat[idx]
        nested_hat_vec[test_index] = nested_hat[idx]
        px_hat_vec[test_index] = px_hat[idx]
        pmx_hat_vec[test_index] = pmx_hat[idx]

    return {
        "u_hat": u_hat,
        "w_hat": w_hat,
        "ymx_hat": ymx_hat_vec,
        "nested_hat": nested_hat_vec,
        "px_hat": px_hat_vec,
        "pmx_hat": pmx_hat_vec,
    }


def med_orth_manual(residuals, d, m, target, treatment_level, mediation_level):
    treated = d == treatment_level
    if target == "potential":
        u_hat = residuals["u_hat"]
        yx_hat = residuals["yx_hat"]
        px_hat = residuals["px_hat"]

        psi_b = np.multiply(np.divide(treated, px_hat), u_hat) + yx_hat
    else:
        u_hat = residuals["u_hat"]
        w_hat = residuals["w_hat"]
        nested_hat = residuals["nested_hat"]
        px_hat = residuals["px_hat"]
        pmx_hat = residuals["pmx_hat"]

        t1 = np.multiply(
            np.multiply(np.divide(treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)),
            u_hat,
        )
        t2 = np.multiply(np.divide(1.0 - treated, 1.0 - px_hat), w_hat)
        psi_b = t1 + t2 + nested_hat

    return np.mean(psi_b)


def var_med_manual(theta, residuals, d, m, target, treatment_level, mediation_level):
    treated = d == treatment_level
    n_obs = len(d)

    if target == "potential":
        u_hat = residuals["u_hat"]
        yx_hat = residuals["yx_hat"]
        px_hat = residuals["px_hat"]

        var = 1 / n_obs * np.mean(np.power(yx_hat + np.divide(np.multiply(treated, u_hat), px_hat) - theta, 2))
    else:
        u_hat = residuals["u_hat"]
        w_hat = residuals["w_hat"]
        nested_hat = residuals["nested_hat"]
        px_hat = residuals["px_hat"]
        pmx_hat = residuals["pmx_hat"]

        var = np.divide(
            np.mean(
                np.power(
                    np.multiply(np.multiply(np.divide(treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)), u_hat)
                    + np.multiply(np.divide(1.0 - treated, 1.0 - px_hat), w_hat)
                    + nested_hat
                    - theta,
                    2,
                )
            ),
            n_obs,
        )

    return var


def boot_med_manual(
    thetas,
    ses,
    all_preds,
    all_smpls,
    y,
    d,
    m,
    target,
    treatment_level,
    mediation_level,
    bootstrap="normal",
    n_rep_boot=500,
):
    n_rep = len(thetas)
    boot_theta = np.zeros(shape=(n_rep, n_rep_boot))
    boot_t_stat = np.zeros(shape=(n_rep, n_rep_boot))

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        preds = {k: v[i_rep] for k, v in all_preds.items()}

        n_obs = len(y)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)

        boot_theta[i_rep, :], boot_t_stat[i_rep, :] = boot_med_single_split_manual(
            thetas[i_rep],
            y,
            d,
            m,
            preds,
            smpls,
            target,
            treatment_level,
            mediation_level,
            ses[i_rep],
            weights,
            n_rep_boot,
        )

    return boot_theta, boot_t_stat


# TODO: Check if this function is correct
def boot_med_single_split_manual(
    theta,
    y,
    d,
    m,
    preds,
    smpls,
    target,
    treatment_level,
    mediation_level,
    se,
    weights,
    n_rep_boot,
):
    residuals = compute_residuals_manual(
        y,
        d,
        m,
        preds,
        smpls,
        target,
        treatment_level,
        mediation_level,
        trimming_threshold=1e-2,
    )

    treated = d == treatment_level
    if target == "potential":
        u_hat = residuals["u_hat"]
        yx_hat = residuals["yx_hat"]
        px_hat = residuals["px_hat"]

        psi_b = np.multiply(np.divide(treated, px_hat), u_hat) + yx_hat
    elif target == "counterfactual":
        u_hat = residuals["u_hat"]
        w_hat = residuals["w_hat"]
        nested_hat = residuals["nested_hat"]
        px_hat = residuals["px_hat"]
        pmx_hat = residuals["pmx_hat"]

        t1 = np.multiply(
            np.multiply(np.divide(treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)),
            u_hat,
        )
        t2 = np.multiply(np.divide(1.0 - treated, 1.0 - px_hat), w_hat)
        psi_b = t1 + t2 + nested_hat
    else:
        raise ValueError(f"Invalid target: {target}")

    psi_a = -1.0
    J = np.mean(psi_a)
    psi = psi_a * theta + psi_b

    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting=True)

    boot_theta = theta + boot_t_stat * se

    return boot_theta, boot_t_stat
