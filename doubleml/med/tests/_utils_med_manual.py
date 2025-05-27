import numpy as np
from sklearn.base import clone, is_classifier

from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search
from ...tests._utils_boot import boot_manual, draw_weights
from ...utils._estimation import _get_cond_smpls, _get_cond_smpls_2d
from ...utils._propensity_score import _normalize_ipw


def _get_nuisance_params(score, score_type, learner_g, learner_m, learner_med=None, learner_nest=None):
    valid_types_score = ["efficient", "ipw"]
    valid_scores = ["Y(d, M(d))", "Y(d, M(1-d))"]

    if score not in valid_scores:
        raise ValueError("Defined score function is not implemented")
    if score_type not in valid_types_score:
        raise ValueError("Type of score function not implemented")

    # TODO: samples for ipw are only placeholders. need to define way to separate samples as needed.
    nuisance_learners_dict = {
        "ipw": {
            "Y(d, M(d))": {
                "learners_names": ["ml_g_d", "ml_m"],
                "learners": [clone(learner_g), clone(learner_m)],
            },
            "Y(d, M(1-d))": {
                "learners_names": ["ml_g_d", "ml_g_d_1md", "ml_m", "ml_m_med"],
                "learners": [clone(learner_g), clone(learner_nest), clone(learner_m), clone(learner_m)],
            },
        },
        "efficient": {
            "Y(d, M(d))": {
                "learners_names": ["ml_g_d", "ml_m"],
                "learners": [clone(learner_g), clone(learner_m)],
            },
            "Y(d, M(1-d))": {
                "learners_names": ["ml_g_d_med_d", "ml_g_d_med_1md", "ml_m", "ml_med_d", "ml_med_1md"],
                "learners": [clone(learner_g), clone(learner_g), clone(learner_m), clone(learner_med), clone(learner_med)],
            },
        },
    }

    learners_names = nuisance_learners_dict[score_type][score]["learners_names"]
    learners = nuisance_learners_dict[score_type][score]["learners"]
    return learners_names, learners


def _get_data_params(
    y,
    d,
    m,
    x,
    type_score,
    score,
    smpls,
    treated,
    mediated,
):

    train_cond_1md, train_cond_d = _get_cond_smpls(smpls, treated)
    train_cond_1md_1md, train_cond_1md_d, train_cond_d_1md, train_cond_d_d = _get_cond_smpls_2d(smpls, treated, mediated)

    xm = np.concat((x, m), axis=1)

    train_cond_list = list()
    ind_var = list()
    dep_var = list()
    if type_score == "ipw":
        if score == "Y(d,M(d))":
            train_cond_list = [train_cond_1md, smpls]
            ind_var = [y, d]
            dep_var = [x, x]
        elif score == "Y(d,M(1-d))":
            train_cond_list = [train_cond_d_d, train_cond_d_1md, smpls, train_cond_1md, train_cond_d]
            ind_var = [y, y, d, d]
            dep_var = [x, xm, x, xm]
    elif type_score == "efficient":
        if score == "Y(d,M(d))":
            train_cond_list = [train_cond_1md, smpls]
            ind_var = [y, d]
            dep_var = [x, x]
        elif score == "Y(d,M(1-d))":
            train_cond_list = [train_cond_1md_1md, train_cond_1md_d, smpls, train_cond_1md, train_cond_d]
            ind_var = [y, y, d, m, m]
            dep_var = [x, x, x, x, x]

        data_params = {"train_cond_list": train_cond_list, "independent_var": ind_var, "depependent_var": dep_var}
    return data_params


def _get_learner_params(
    score,
    type_score,
    g_d0_params=None,
    g_d_params=None,
    g_d_1md_params=None,
    g_d_med_d_params=None,
    g_d_med_1md_params=None,
    m_params=None,
    med_d_params=None,
    med_1md_params=None,
    m_med_params=None,
):

    learner_params = list()
    if type_score == "ipw":
        if score == "Y(d,M(d))":
            learner_params = [g_d_params, m_params]
        elif score == "Y(d,M(1-d))":
            learner_params = [g_d_params, g_d_1md_params, m_params, m_med_params]
    elif type_score == "efficient":
        if score == "Y(d,M(d))":
            learner_params = [g_d_params, m_params]
        elif score == "Y(d,M(1-d))":
            learner_params = [g_d_med_d_params, g_d_med_1md_params, m_params, med_d_params, med_1md_params]

    return learner_params


def fit_med(
    y,
    x,
    d,
    m,
    learner_g,
    learner_m,
    learner_med,
    learner_nest,
    all_smpls,
    score,
    type_score,
    n_rep=1,
    g_d0_params=None,
    g_d1_params=None,
    g_d0_d0_params=None,
    g_d0_d1_params=None,
    g_d1_d0_params=None,
    g_d1_d1_params=None,
    g_d0_med0_params=None,
    g_d0_med1_params=None,
    g_d1_med0_params=None,
    g_d1_med1_params=None,
    m_params=None,
    med_d0_params=None,
    med_d1_params=None,
    m_med_params=None,
    normalize_ipw=False,
    trimming_threshold=1e-2,
):
    n_obs = len(y)
    treated = d == 1
    mediated = m == 1
    #    mediation_level = 0
    #    treatment_level = 0

    #    if score == "Y(d, M(1-d))":
    #        mediation_level = 1
    #    elif score == "Y(1, M(0))":
    #        treatment_level = 1
    #    elif score == "Y(1, M(1))":
    #        treatment_level = 1
    #        mediation_level = 1

    # Get the name of the learners and a learner object for the given score function.
    learners_names, learners = _get_nuisance_params(score, type_score, learner_g, learner_m, learner_med, learner_nest)
    learner_params = _get_learner_params(
        g_d0_params,
        g_d1_params,
        g_d0_d0_params,
        g_d0_d1_params,
        g_d1_d0_params,
        g_d1_d1_params,
        g_d0_med0_params,
        g_d0_med1_params,
        g_d1_med0_params,
        g_d1_med1_params,
        m_params,
        med_d0_params,
        med_d1_params,
        m_med_params,
    )
    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    # Create a dictionnary to store each learners predictions.
    learner_preds = dict()
    for learner in learners_names:
        x = learner.split("_", 1)
        learner_preds.setdefault("all_" + x[1] + "hat", list())

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        data_params = _get_data_params(y, d, m, x, type_score, score, smpls, treated, mediated)
        preds = fit_nuisance_med(
            smpls,
            data_params,
            treated,
            mediated,
            learners,
            learners_names,
            score,
            type_score,
            learner_params,
        )

        # Update the list of predictions for each learner.
        #        pred_keys = learner_preds.keys()
        for index, (key, value) in enumerate(learner_preds):
            cur_list = learner_preds[key]
            learner_preds.update({key: cur_list.append(preds[index])})

        thetas[i_rep], ses[i_rep] = med_dml2(y, x, d, m, treated, mediated, preds, smpls, score, normalize_ipw)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {
        "theta": theta,
        "se": se,
        "thetas": thetas,
        "ses": ses,
    }

    res.update(learner_preds)

    return res


def fit_nuisance_med(
    smpls,
    data_params,
    treated,
    mediated,
    learners,
    learners_names,
    score,
    type_score,
    learner_params,
    trimming_threshold=1e-12,
):
    pred_list = list()

    for index in range(learners):
        train_cond, ind_var, dep_var = data_params.values
        pred_list.append(
            _fit_predict(
                ind_var[index],
                dep_var[index],
                smpls,
                learner_params[index],
                learners[index],
                train_cond[0],
            )
        )

    return pred_list


def _fit_predict(ind_var, dep_var, smpls, learner_params, learner, train_cond, trimming_threshold=1e-12):
    if is_classifier(learner):
        learner_pred = fit_predict_proba(
            ind_var, dep_var, learner, learner_params, smpls, trimming_threshold, train_cond=train_cond
        )
    else:
        learner_pred = fit_predict(ind_var, dep_var, learner, learner_params, smpls, train_cond=train_cond)
    return learner_pred


def compute_residuals(y, type_score, score, preds, smpls):
    if score == "Y(d, M(d))" or score == "Y(1, M(1))":
        res = _compute_potential_residuals(type_score, smpls, y, preds)
    elif score == "Y(d, M(1-d))" or score == "Y(1, M(0))":
        res = _compute_counterfactual_residuals(type_score, smpls, y, preds)

    # TODO: Check this propensity funct.
    # _check_is_propensity(m_hat, "learner_m", "ml_m", smpls, eps=1e-12)
    return res


def _compute_potential_residuals(type_score, smpls, y, preds):
    g_hat = np.full_like(y, np.nan, dtype="float64")
    m_hat = np.full_like(y, np.nan, dtype="float64")
    g_d_hat = np.full_like(y, np.nan, dtype="float64")

    for idx, (_, test_index) in enumerate(smpls):
        g_hat[test_index] = y[test_index] - preds[0][idx]
        m_hat[test_index] = preds[1][idx]
        g_d_hat[test_index] = preds[0][idx]
    return g_hat, m_hat, g_d_hat


def _compute_counterfactual_residuals(type_score, smpls, y, preds):
    u_hat = np.full_like(y, np.nan, dtype="float64")
    v_hat = np.full_like(y, np.nan, dtype="float64")
    w_hat = np.full_like(y, np.nan, dtype="float64")
    m_hat = np.full_like(y, np.nan, dtype="float64")
    m_med_hat = np.full_like(y, np.nan, dtype="float64")
    m_c_med_hat = np.full_like(y, np.nan, dtype="float64")

    # TODO: Don't want to recompute eta but have to use it to compute scores, so how could I do it?
    # TODO: Maybe have to swap the [n] and [idx] in preds[n][idx]
    for idx, (_, test_index) in enumerate(smpls):
        if type_score == "ipw":
            w_hat[test_index] = preds[2][idx]
            m_med_hat[test_index] = preds[3][idx]
            m_c_med_hat[test_index] = 1 - m_med_hat[test_index]
        elif type_score == "efficient":
            w_hat[test_index] = preds[0][idx] * preds[5][idx] + preds[1][idx] * (1 - preds[5][idx])
            m_med_hat[test_index] = preds[4][idx]
            m_c_med_hat[test_index] = preds[3][idx]

        u_hat[test_index] = y[test_index] - preds[0][idx]
        v_hat[test_index] = preds[0][idx] - w_hat[test_index][idx]
        m_hat = preds[2][idx]

    return u_hat, v_hat, w_hat, m_hat, m_med_hat, m_c_med_hat


def med_dml2(y, d, type_score, score, treated, treatment_level, preds, smpls, normalize_ipw):
    n_obs = len(y)
    data_treatment_level = d == treatment_level
    residuals = compute_residuals(y, type_score, score, preds, smpls, data_treatment_level)

    # TODO: Check how to normalize_ipw easier:
    #    if normalize_ipw:
    #        m_hat_adj = _normalize_ipw(m_hat, treated)
    #    else:
    #        m_hat_adj = m_hat

    theta_hat = med_orth(type_score, score, preds, smpls, y, d, treatment_level)

    se = np.sqrt(var_med(theta_hat, residuals, treated, score, n_obs, data_treatment_level))

    return theta_hat, se


def med_orth(type_score, residuals, score, preds, smpls, y, d, data_treatment_level):
    if score == "Y(d, M(d))" or score == "Y(d, M(1-d))":
        g_hat, m_hat, g_d_hat = residuals
        res = np.mean(g_d_hat + np.divide(np.multiply(data_treatment_level, g_hat), m_hat))

    elif score == "Y(1, M(0))" or score == "Y(1, M(1))":
        u_hat, v_hat, w_hat, m_hat, m_med_hat, m_c_med_hat = residuals
        res = (
            np.mean(
                np.multiply(
                    np.divide(np.multiply(np.multiply(data_treatment_level, d), m_c_med_hat), np.multiply(m_hat, m_med_hat)),
                    u_hat,
                )
            )
            + np.divide(np.multiply(np.multiply(d, (1 - data_treatment_level)), v_hat), (1 - m_hat))
            + w_hat
        )

    return res


def var_med(theta, d, residuals, treated, score, n_obs, data_treatment_level):
    if score == "Y(d, M(d))" or score == "Y(d, M(1-d))":
        g_hat, m_hat, g_d_hat = residuals
        var = 1 / n_obs * np.mean(np.power(g_d_hat + np.divide(np.multiply(data_treatment_level, g_hat), m_hat) - theta, 2))
    elif score == "Y(1, M(0))" or score == "Y(1, M(1))":
        u_hat, v_hat, w_hat, m_hat, m_med_hat, m_c_med_hat = residuals
        var = (
            np.mean(
                np.multiply(
                    np.divide(np.multiply(np.multiply(data_treatment_level, d), m_c_med_hat), np.multiply(m_hat, m_med_hat)),
                    u_hat,
                )
            )
            + np.divide(np.multiply(np.multiply(d, (1 - data_treatment_level)), v_hat), (1 - m_hat))
            + w_hat
        )

    return var


def boot_med(
    y,
    d,
    treatment_level,
    thetas,
    ses,
    all_g_hat0,
    all_g_hat1,
    all_m_hat,
    all_smpls,
    score,
    bootstrap,
    n_rep_boot,
    n_rep=1,
    normalize_ipw=True,
):
    treated = d == treatment_level
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        n_obs = len(y)

        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_t_stat = boot_med_single_split(
            thetas[i_rep],
            y,
            d,
            treated,
            all_g_hat0[i_rep],
            all_g_hat1[i_rep],
            all_m_hat[i_rep],
            smpls,
            score,
            ses[i_rep],
            weights,
            n_rep_boot,
            normalize_ipw,
        )
        all_boot_t_stat.append(boot_t_stat)

    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_t_stat


def boot_med_single_split(
    theta, y, d, treated, g_hat0_list, g_hat1_list, m_hat_list, smpls, score, se, weights, n_rep_boot, normalize_ipw
):
    _, u_hat1, _, g_hat1, m_hat = compute_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, smpls)

    if normalize_ipw:
        m_hat_adj = _normalize_ipw(m_hat, treated)
    else:
        m_hat_adj = m_hat

    J = -1.0
    psi = g_hat1 + np.divide(np.multiply(treated, u_hat1), m_hat_adj) - theta
    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

    return boot_t_stat


def fit_sensitivity_elements_med(y, d, treatment_level, all_coef, predictions, score, n_rep, normalize_ipw):
    n_treat = 1
    n_obs = len(y)
    treated = d == treatment_level

    sigma2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    nu2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    psi_sigma2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)
    psi_nu2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)

    for i_rep in range(n_rep):
        m_hat = predictions["ml_m"][:, i_rep, 0]
        if normalize_ipw:
            m_hat_adj = _normalize_ipw(m_hat, treated)
        else:
            m_hat_adj = m_hat
        g_hat0 = predictions["ml_g_d_lvl0"][:, i_rep, 0]
        g_hat1 = predictions["ml_g_d_lvl1"][:, i_rep, 0]

        weights = np.ones_like(d)
        weights_bar = np.ones_like(d)

        sigma2_score_element = np.square(y - np.multiply(treated, g_hat1) - np.multiply(1.0 - treated, g_hat0))
        sigma2[0, i_rep, 0] = np.mean(sigma2_score_element)
        psi_sigma2[:, i_rep, 0] = sigma2_score_element - sigma2[0, i_rep, 0]

        # calc m(W,alpha) and Riesz representer
        m_alpha = np.multiply(weights, np.multiply(weights_bar, np.divide(1.0, m_hat_adj)))
        rr = np.multiply(weights_bar, np.divide(treated, m_hat_adj))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2[0, i_rep, 0] = np.mean(nu2_score_element)
        psi_nu2[:, i_rep, 0] = nu2_score_element - nu2[0, i_rep, 0]

    element_dict = {"sigma2": sigma2, "nu2": nu2, "psi_sigma2": psi_sigma2, "psi_nu2": psi_nu2}
    return element_dict


def tune_nuisance_med(y, x, d, treatment_level, ml_g, ml_m, smpls, score, n_folds_tune, param_grid_g, param_grid_m):
    dx = np.column_stack((d, x))
    train_cond0 = np.where(d != treatment_level)[0]
    g0_tune_res = tune_grid_search(y, dx, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond0)

    train_cond1 = np.where(d == treatment_level)[0]
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond1)

    treated = d == treatment_level
    m_tune_res = tune_grid_search(treated, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params
