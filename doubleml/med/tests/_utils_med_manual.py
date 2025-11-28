import copy

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split

from doubleml.tests._utils import fit_predict, fit_predict_proba, tune_grid_search
from doubleml.tests._utils_boot import boot_manual, draw_weights
from doubleml.utils._checks import _check_is_propensity
from doubleml.utils._propensity_score import _normalize_ipw


class ManualMedP:
    def __init__(self, y, x, d, m, learner_g, learner_m, treatment_level, all_smpls, n_rep, normalize_ipw, trimming_threshold):
        self._train_test_samples = None
        self.y = y
        self.x = x
        self.d = d
        self.m = m
        self.learner_g = learner_g
        self.learner_m = learner_m
        self.treatment_level = treatment_level
        self.all_smpls = all_smpls
        self.n_rep = n_rep
        self.normalize_ipw = normalize_ipw
        self.trimming_threshold = trimming_threshold
        self.treated = self.d == self.treatment_level

    def fit_med(
        self,
        g0_params=None,
        g1_params=None,
        px_params=None,
    ):
        n_obs = len(self.y)

        thetas = np.zeros(self.n_rep)
        ses = np.zeros(self.n_rep)
        all_g_hat0 = list()
        all_g_hat1 = list()
        all_m_hat = list()

        for i_rep in range(self.n_rep):
            smpls = self.all_smpls[i_rep]
            g_hat0, g_hat1, m_hat = self.fit_nuisance_med(
                smpls,
                g0_params=g0_params,
                g1_params=g1_params,
                px_params=px_params,
            )

            all_g_hat0.append(g_hat0)
            all_g_hat1.append(g_hat1)
            all_m_hat.append(m_hat)

            thetas[i_rep], ses[i_rep] = self.med_dml2(
                smpls,
                g_hat0_list=g_hat0,
                g_hat1_list=g_hat1,
                m_hat_list=m_hat,
            )

        theta = np.median(thetas)
        se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

        res = {
            "theta": theta,
            "se": se,
            "thetas": thetas,
            "ses": ses,
            "all_g_hat0": all_g_hat0,
            "all_g_hat1": all_g_hat1,
            "all_m_hat": all_m_hat,
        }

        return res

    def fit_nuisance_med(
        self,
        smpls,
        g0_params=None,
        g1_params=None,
        px_params=None,
    ):
        ml_g0 = clone(self.learner_g)
        ml_g1 = clone(self.learner_g)
        dx = np.column_stack((self.d, self.x))

        train_cond_d0 = np.where(self.treated == 0)[0]
        if is_classifier(self.learner_g):
            g_hat0_list = fit_predict_proba(self.y, dx, ml_g0, g0_params, smpls, train_cond=train_cond_d0)
        else:
            g_hat0_list = fit_predict(self.y, dx, ml_g0, g0_params, smpls, train_cond=train_cond_d0)

        train_cond_d1 = np.where(self.treated == 1)[0]
        if is_classifier(self.learner_g):
            g_hat1_list = fit_predict_proba(self.y, self.x, ml_g1, g1_params, smpls, train_cond=train_cond_d1)
        else:
            g_hat1_list = fit_predict(self.y, self.x, ml_g1, g1_params, smpls, train_cond=train_cond_d1)

        ml_m = clone(self.learner_m)
        ###Debugging
        # Turned off trimming_threshold for debugging. Turn it back on.
        m_hat_list = fit_predict_proba(self.treated, self.x, ml_m, px_params, smpls, trimming_threshold=0)
        ###
        return g_hat0_list, g_hat1_list, m_hat_list

    def compute_residuals(
        self,
        smpls,
        g_hat0_list,
        g_hat1_list,
        m_hat_list,
    ):
        u_hat0 = np.full_like(self.y, np.nan, dtype="float64")
        u_hat1 = np.full_like(self.y, np.nan, dtype="float64")
        g_hat0 = np.full_like(self.y, np.nan, dtype="float64")
        g_hat1 = np.full_like(self.y, np.nan, dtype="float64")
        m_hat = np.full_like(self.y, np.nan, dtype="float64")
        for idx, (_, test_index) in enumerate(smpls):
            u_hat0[test_index] = self.y[test_index] - g_hat0_list[idx]
            u_hat1[test_index] = self.y[test_index] - g_hat1_list[idx]
            g_hat0[test_index] = g_hat0_list[idx]
            g_hat1[test_index] = g_hat1_list[idx]
            m_hat[test_index] = m_hat_list[idx]

        _check_is_propensity(m_hat, "learner_m", "ml_m", smpls, eps=1e-12)
        return u_hat0, u_hat1, g_hat0, g_hat1, m_hat

    def med_dml2(
        self,
        smpls,
        g_hat0_list,
        g_hat1_list,
        m_hat_list,
    ):
        n_obs = len(self.y)
        u_hat0, u_hat1, g_hat0, g_hat1, m_hat = self.compute_residuals(
            smpls,
            g_hat0_list,
            g_hat1_list,
            m_hat_list,
        )
        #### Debugging
        # if self.normalize_ipw:
        #   m_hat_adj = _normalize_ipw(m_hat, self.treated)
        # else:
        #    m_hat_adj = m_hat
        m_hat_adj = m_hat
        ####

        theta_hat = self.med_orth(
            g_hat0=g_hat0,
            g_hat1=g_hat1,
            m_hat=m_hat_adj,
            u_hat1=u_hat1,
        )

        se = np.sqrt(self.var_med(theta_hat, g_hat0, g_hat1, m_hat_adj, u_hat0, u_hat1, n_obs))

        return theta_hat, se

    def med_orth(
        self,
        g_hat0,
        g_hat1,
        m_hat,
        u_hat1,
    ):
        res = np.mean(g_hat1 + np.multiply(np.divide(self.treated, m_hat), u_hat1))
        return res

    def var_med(self, theta, g_hat0, g_hat1, m_hat, u_hat0, u_hat1, n_obs):
        var = 1 / n_obs * np.mean(np.power(g_hat1 + np.divide(np.multiply(self.treated, u_hat1), m_hat) - theta, 2))
        return var

    def boot_med(
        self,
        thetas,
        ses,
        all_g_hat0,
        all_g_hat1,
        all_m_hat,
        bootstrap,
        n_rep_boot,
    ):

        all_boot_t_stat = list()
        for i_rep in range(self.n_rep):
            smpls = self.all_smpls[i_rep]
            n_obs = len(self.y)

            weights = draw_weights(bootstrap, n_rep_boot, n_obs)
            boot_t_stat = self.boot_med_single_split(
                thetas[i_rep],
                all_g_hat0[i_rep],
                all_g_hat1[i_rep],
                all_m_hat[i_rep],
                smpls,
                ses[i_rep],
                weights,
                n_rep_boot,
            )
            all_boot_t_stat.append(boot_t_stat)

        boot_t_stat = np.hstack(all_boot_t_stat)

        return boot_t_stat

    def boot_med_single_split(
        self,
        theta,
        g_hat0_list,
        g_hat1_list,
        m_hat_list,
        smpls,
        se,
        weights,
        n_rep_boot,
    ):
        _, u_hat1, _, g_hat1, m_hat = self.compute_residuals(
            smpls,
            g_hat0_list,
            g_hat1_list,
            m_hat_list,
        )
        ###Debugging
        # if self.normalize_ipw:
        #    m_hat_adj = _normalize_ipw(m_hat, self.treated)
        # else:
        #    m_hat_adj = m_hat
        m_hat_adj = m_hat
        ###
        J = -1.0
        psi = g_hat1 + np.divide(np.multiply(self.treated, u_hat1), m_hat_adj) - theta
        boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

        return boot_t_stat

    def fit_sensitivity_elements_med(
        self,
        all_coef,
        predictions,
    ):
        n_treat = 1
        n_obs = len(self.y)

        sigma2 = np.full(shape=(1, self.n_rep, n_treat), fill_value=np.nan)
        nu2 = np.full(shape=(1, self.n_rep, n_treat), fill_value=np.nan)
        psi_sigma2 = np.full(shape=(n_obs, self.n_rep, n_treat), fill_value=np.nan)
        psi_nu2 = np.full(shape=(n_obs, self.n_rep, n_treat), fill_value=np.nan)

        for i_rep in range(self.n_rep):
            m_hat = predictions["ml_m"][:, i_rep, 0]
            if self.normalize_ipw:
                m_hat_adj = _normalize_ipw(m_hat, self.treated)
            else:
                m_hat_adj = m_hat
            g_hat0 = predictions["ml_g_d0"][:, i_rep, 0]
            g_hat1 = predictions["ml_yx"][:, i_rep, 0]

            weights = np.ones_like()
            weights_bar = np.ones_like()

            sigma2_score_element = np.square(
                self.y - np.multiply(self.treated, g_hat1) - np.multiply(1.0 - self.treated, g_hat0)
            )
            sigma2[0, i_rep, 0] = np.mean(sigma2_score_element)
            psi_sigma2[:, i_rep, 0] = sigma2_score_element - sigma2[0, i_rep, 0]

            # calc m(W,alpha) and Riesz representer
            m_alpha = np.multiply(weights, np.multiply(weights_bar, np.divide(1.0, m_hat_adj)))
            rr = np.multiply(weights_bar, np.divide(self.treated, m_hat_adj))

            nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
            nu2[0, i_rep, 0] = np.mean(nu2_score_element)
            psi_nu2[:, i_rep, 0] = nu2_score_element - nu2[0, i_rep, 0]

        element_dict = {"sigma2": sigma2, "nu2": nu2, "psi_sigma2": psi_sigma2, "psi_nu2": psi_nu2}
        return element_dict

    def tune_nuisance_med(
        self,
        ml_g,
        ml_m,
        smpls,
        n_folds_tune,
        param_grid_g,
        param_grid_m,
    ):
        dx = np.column_stack((self.d, self.x))
        train_cond_d0 = np.where(self.d != self.treatment_level)[0]
        g0_tune_res = tune_grid_search(self.y, dx, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond_d0)

        train_cond_d1 = np.where(self.d == self.treatment_level)[0]
        g1_tune_res = tune_grid_search(self.y, self.x, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond_d1)

        m_tune_res = tune_grid_search(self.treated, self.x, ml_m, smpls, param_grid_m, n_folds_tune)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        return g0_best_params, g1_best_params, m_best_params


class ManualMedC:
    def fit_med(
        y,
        x,
        d,
        m,
        learner_g,
        learner_m,
        learner_med,
        treatment_level,
        all_smpls,
        g_d1_m1_params=None,
        g_d1_m0_params=None,
        med_d1_params=None,
        med_d0_params=None,
        px_params=None,
        n_rep=1,
        normalize_ipw=False,
        trimming_threshold=1e-2,
    ):
        n_obs = len(y)
        treated = d == treatment_level
        mediated = m == treatment_level

        thetas = np.zeros(n_rep)
        ses = np.zeros(n_rep)

        all_g_d1_m1_hat = list()
        all_g_d1_m0_hat = list()
        all_med_d1_hat = list()
        all_med_d0_hat = list()
        all_m_hat = list()

        for i_rep in range(n_rep):
            smpls = all_smpls[i_rep]
            g_d1_m1_hat, g_d1_m0_hat, med_d1_hat, med_d0_hat, m_hat = ManualMedC.fit_nuisance_med(
                y=y,
                x=x,
                d=d,
                m=m,
                treated=treated,
                mediated=mediated,
                learner_g=learner_g,
                learner_m=learner_m,
                learner_med=learner_med,
                smpls=smpls,
                g_d1_m1_params=g_d1_m1_params,
                g_d1_m0_params=g_d1_m0_params,
                med_d1_params=med_d1_params,
                med_d0_params=med_d0_params,
                px_params=px_params,
                trimming_threshold=trimming_threshold,
            )

            all_g_d1_m1_hat.append(g_d1_m1_hat)
            all_g_d1_m0_hat.append(med_d0_hat)
            all_med_d1_hat.append(med_d1_hat)
            all_med_d0_hat.append(med_d0_hat)
            all_m_hat.append(m_hat)

            thetas[i_rep], ses[i_rep] = ManualMedC.med_dml2(
                y=y,
                x=x,
                d=d,
                m=m,
                treated=treated,
                mediated=mediated,
                g_d1_m1_hat=g_d1_m1_hat,
                med_d0_hat=med_d0_hat,
                med_d1_hat=med_d1_hat,
                m_hat=m_hat,
                smpls=smpls,
                normalize_ipw=normalize_ipw,
            )

        theta = np.median(thetas)
        se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

        res = {
            "theta": theta,
            "se": se,
            "thetas": thetas,
            "ses": ses,
            "all_g_d1_m1_hat": all_g_d1_m1_hat,
            "all_g_d1_m0_hat": all_g_d1_m0_hat,
            "all_med_d1_hat": all_med_d1_hat,
            "all_med_d0_hat": all_med_d0_hat,
            "all_m_hat": all_m_hat,
        }

        return res

    def fit_nuisance_med(
        y,
        x,
        treated,
        mediated,
        learner_g,
        learner_m,
        smpls,
        learner_med,
        g_d1_m1_params=None,
        g_d1_m0_params=None,
        med_d1_params=None,
        med_d0_params=None,
        px_params=None,
        trimming_threshold=1e-12,
    ):

        ml_g_d1_m1 = clone(learner_g)
        ml_g_d1_m0 = clone(learner_g)
        ml_med_d1 = clone(learner_med)
        ml_med_d0 = clone(learner_med)

        # TODO: Find way to select the intersection between m and d for train_cond_xy,
        # where x, y in [0, 1] and xy = cartesian_product(x, y)
        # For 1-dimensional d and m.

        train_cond_d1_m0 = np.where((treated == 1) & (mediated == 0))[0]
        train_cond_d1_m1 = np.where((treated == 1) & (mediated == 1))[0]
        if is_classifier(learner_g):
            g_d1_m1_hat_list = fit_predict_proba(y, x, ml_g_d1_m1, g_d1_m1_params, smpls, train_cond=train_cond_d1_m1)
            g_d1_m0_hat_list = fit_predict_proba(y, x, ml_g_d1_m0, g_d1_m0_params, smpls, train_cond=train_cond_d1_m0)
        else:
            g_d1_m1_hat_list = fit_predict(y, x, ml_g_d1_m1, g_d1_m1_params, smpls, train_cond=train_cond_d1_m1)
            g_d1_m0_hat_list = fit_predict(y, x, ml_g_d1_m0, g_d1_m0_params, smpls, train_cond=train_cond_d1_m0)

        train_cond_d0 = np.where(treated == 0)[0]
        train_cond_d1 = np.where(treated == 1)[0]
        if is_classifier(learner_med):
            med_d1_hat_list = fit_predict_proba(y, x, ml_med_d1, med_d1_params, smpls, train_cond=train_cond_d1)
            med_d0_hat_list = fit_predict_proba(y, x, ml_med_d0, med_d0_params, smpls, train_cond=train_cond_d0)
        else:
            med_d1_hat_list = fit_predict(y, x, ml_med_d1, med_d1_params, smpls, train_cond=train_cond_d1)
            med_d0_hat_list = fit_predict(y, x, ml_med_d0, med_d0_params, smpls, train_cond=train_cond_d0)

        ml_m = clone(learner_m)
        # TODO: reset trimming_threshold to parameter
        m_hat_list = fit_predict_proba(treated, x, ml_m, px_params, smpls, trimming_threshold=0)

        return g_d1_m1_hat_list, g_d1_m0_hat_list, med_d1_hat_list, med_d0_hat_list, m_hat_list

    def compute_residuals(
        y,
        m,
        smpls,
        treatment_level,
        m_hat_list,
        g_d1_m1_hat_list,
        g_d1_m0_hat_list,
        med_d1_hat_list,
        med_d0_hat_list,
    ):

        u_hat = np.full_like(y, np.nan, dtype="float64")
        w_hat = np.full_like(y, np.nan, dtype="float64")
        g_d1_m0_hat = np.full_like(y, np.nan, dtype="float64")
        g_d1_m1_hat = np.full_like(y, np.nan, dtype="float64")
        med_d0_hat = np.full_like(y, np.nan, dtype="float64")
        med_d1_hat = np.full_like(y, np.nan, dtype="float64")
        m_hat = np.full_like(y, np.nan, dtype="float64")

        for idx, (_, test_index) in enumerate(smpls):
            if treatment_level == 1:
                y_d_m_hat = g_d1_m1_hat_list[idx] * med_d0_hat_list[idx] + g_d1_m0_hat_list[idx] * (1 - med_d0_hat_list[idx])
                g_d_hat = m[test_index] * g_d1_m1_hat_list[idx] + (1.0 - m[test_index]) * g_d1_m0_hat_list[idx]
            else:
                y_d_m_hat = g_d1_m1_hat_list[idx] * (1 - med_d1_hat_list[idx]) + g_d1_m0_hat_list[idx] * med_d1_hat_list[idx]
                g_d_hat = m[test_index] * g_d1_m0_hat_list[idx] + (1.0 - m[test_index]) * g_d1_m1_hat_list[idx]

            g_d1_m0_hat[test_index] = g_d1_m0_hat_list[idx]
            g_d1_m1_hat[test_index] = g_d1_m1_hat_list[idx]
            med_d0_hat[test_index] = med_d0_hat_list[idx]
            med_d1_hat[test_index] = med_d1_hat_list[idx]
            m_hat[test_index] = m_hat_list[idx]
            u_hat[test_index] = y[test_index] - g_d_hat[idx]
            w_hat[test_index] = g_d_hat[idx] - y_d_m_hat[idx]

        _check_is_propensity(m_hat, "learner_m", "ml_m", smpls, eps=1e-12)
        return g_d1_m0_hat, g_d1_m1_hat, med_d0_hat, med_d1_hat, m_hat, u_hat, w_hat, y_d_m_hat

    # TODO: Probably add method to get adjusted m_hat
    def med_dml2(
        y,
        treated,
        smpls,
        m_hat_list,
        g_d1_m0_hat_list,
        g_d1_m1_hat_list,
        med_d0_hat_list,
        med_d1_hat_list,
    ):
        n_obs = len(y)

        g_d1_m0_hat, g_d1_m1_hat, med_d0_hat, med_d1_hat, m_hat, u_hat, w_hat, y_d_m_hat = ManualMedC.compute_residuals(
            y=y,
            g_d1_m0_hat_list=g_d1_m0_hat_list,
            g_d1_m1_hat_list=g_d1_m1_hat_list,
            med_d0_hat_list=med_d0_hat_list,
            med_d1_hat_list=med_d1_hat_list,
            m_hat_list=m_hat_list,
            smpls=smpls,
        )
        theta_hat = ManualMedC.med_orth(
            treated=treated,
            u_hat=u_hat,
            w_hat=w_hat,
            med_d0_hat=med_d0_hat,
            med_d1_hat=med_d1_hat,
            y_d_m_hat=y_d_m_hat,
            m_hat=m_hat,
        )

        se = np.sqrt(
            ManualMedC.var_med(
                theta_hat=theta_hat,
                med_d0_hat=med_d0_hat,
                med_d1_hat=med_d1_hat,
                m_hat=m_hat,
                u_hat=u_hat,
                w_hat=w_hat,
                treated=treated,
                n_obs=n_obs,
            )
        )

        return theta_hat, se

    def med_orth(
        treated,
        u_hat,
        w_hat,
        med_d0_hat,
        med_d1_hat,
        y_d_m_hat,
        m_hat,
    ):
        res = np.mean(
            np.multiply(np.multiply(np.divide(treated, m_hat), np.divide(med_d0_hat, med_d1_hat)), u_hat)
            + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
            + y_d_m_hat
        )
        return res

    def var_med(theta, treated, n_obs, m_hat, u_hat, w_hat, med_d0_hat, med_d1_hat, y_d_m_hat):
        var = (
            1
            / n_obs
            * np.mean(
                np.power(
                    np.multiply(np.multiply(np.divide(treated, m_hat), np.divide(med_d0_hat, med_d1_hat)), u_hat)
                    + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
                    + y_d_m_hat
                    - theta,
                    2,
                )
            )
        )
        return var

    # TODO: Continue here (2025.08.07)
    def boot_med(
        y,
        d,
        m,
        treatment_level,
        mediation_level,
        thetas,
        ses,
        all_g_d1_m1_hat,
        all_g_d1_m0_hat,
        all_med_d1_hat,
        all_med_d0_hat,
        all_m_hat,
        all_smpls,
        bootstrap,
        n_rep_boot,
        n_rep=1,
        normalize_ipw=True,
    ):
        treated = d == treatment_level
        mediated = m == mediation_level

        all_boot_t_stat = list()
        for i_rep in range(n_rep):
            smpls = all_smpls[i_rep]
            n_obs = len(y)

            weights = draw_weights(bootstrap, n_rep_boot, n_obs)
            boot_t_stat = ManualMedC.boot_med_single_split(
                thetas[i_rep],
                y,
                d,
                treated,
                mediated,
                all_g_d1_m1_hat[i_rep],
                all_g_d1_m0_hat[i_rep],
                all_med_d1_hat[i_rep],
                all_med_d0_hat[i_rep],
                all_m_hat[i_rep],
                smpls,
                ses[i_rep],
                weights,
                n_rep_boot,
                normalize_ipw,
            )
            all_boot_t_stat.append(boot_t_stat)

        boot_t_stat = np.hstack(all_boot_t_stat)

        return boot_t_stat

    def boot_med_single_split(
        theta,
        y,
        treated,
        m_hat_list,
        g_d1_m0_hat_list,
        g_d1_m1_hat_list,
        med_d0_hat_list,
        med_d1_hat_list,
        smpls,
        se,
        weights,
        n_rep_boot,
        normalize_ipw,
    ):
        g_d1_m0_hat, g_d1_m1_hat, med_d0_hat, med_d1_hat, m_hat, u_hat, w_hat, y_d_m_hat = ManualMedC.compute_residuals(
            y, m_hat_list, g_d1_m0_hat_list, g_d1_m1_hat_list, med_d0_hat_list, med_d1_hat_list, smpls
        )

        #    if normalize_ipw:
        #        m_hat_adj = _normalize_ipw(m_hat, treated)
        #    else:
        #        m_hat_adj = m_hat

        J = -1.0
        psi = (
            np.mean(
                np.multiply(np.multiply(np.divide(treated, m_hat), np.divide(med_d0_hat, med_d1_hat)), u_hat)
                + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
                + y_d_m_hat
            )
            - theta
        )
        boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

        return boot_t_stat

    def fit_sensitivity_elements_med(y, d, treatment_level, all_coef, predictions, n_rep, normalize_ipw):
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
            g_hat0 = predictions["ml_g_d0"][:, i_rep, 0]
            g_hat1 = predictions["ml_g_d1"][:, i_rep, 0]

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

    def tune_nuisance_med(
        y,
        x,
        d,
        treatment_level,
        ml_g,
        ml_m,
        smpls,
        n_folds_tune,
        param_grid_g,
        param_grid_m,
    ):
        dx = np.column_stack((d, x))
        train_cond_d0 = np.where(d != treatment_level)[0]
        g0_tune_res = tune_grid_search(y, dx, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond_d0)

        train_cond_d1 = np.where(d == treatment_level)[0]
        g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune, train_cond=train_cond_d1)

        treated = d == treatment_level
        m_tune_res = tune_grid_search(treated, x, ml_m, smpls, param_grid_m, n_folds_tune)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        return g0_best_params, g1_best_params, m_best_params


# TODO: erase smpls_ratio here and in every med file
class ManualMedCAlt:
    def __init__(
        self,
        y,
        x,
        d,
        m,
        learner_yx,
        learner_ymx,
        learner_px,
        learner_pmx,
        learner_nested,
        treatment_level,
        mediation_level,
        all_smpls,
        n_rep,
        normalize_ipw,
        trimming_threshold,
        smpls_ratio,
        n_folds=5,
    ):
        self.n_obs = len(y)
        self.y = y
        self.x = x
        self.d = d
        self.m = m
        self.learner_yx = learner_yx
        self.learner_ymx = learner_ymx
        self.learner_px = learner_px
        self.learner_pmx = learner_pmx
        self.learner_nested = learner_nested
        self.treatment_level = treatment_level
        self.mediation_level = mediation_level
        self.all_smpls = all_smpls
        self.n_rep = n_rep
        self.normalize_ipw = normalize_ipw
        self.trimming_threshold = trimming_threshold
        self.treated = self.d == self.treatment_level
        self.smpls_ratio = smpls_ratio
        self.n_folds = n_folds

    def fit_med(
        self,
        yx_params=None,
        ymx_params=None,
        px_params=None,
        pmx_params=None,
        nested_params=None,
    ):
        thetas = np.zeros(self.n_rep)
        ses = np.zeros(self.n_rep)

        # "ml_yx", "ml_nested", "ml_px", "ml_pmx"
        all_yx_hat = list()
        all_ymx_hat = list()
        all_px_hat = list()
        all_pmx_hat = list()
        all_nested_hat = list()

        for i_rep in range(self.n_rep):
            smpls = self.all_smpls[i_rep]
            yx_hat, ymx_hat, px_hat, pmx_hat, nested_hat = self.fit_nuisance_med(
                smpls=smpls,
                yx_params=yx_params,
                ymx_params=ymx_params,
                px_params=px_params,
                pmx_params=pmx_params,
                nested_params=nested_params,
            )

            all_yx_hat.append(yx_hat)
            all_ymx_hat.append(ymx_hat)
            all_px_hat.append(px_hat)
            all_pmx_hat.append(pmx_hat)
            all_nested_hat.append(nested_hat)

            thetas[i_rep], ses[i_rep] = self.med_dml2(
                smpls=smpls,
                yx_hat_list=yx_hat,
                ymx_hat_list=ymx_hat,
                px_hat_list=px_hat,
                pmx_hat_list=pmx_hat,
                nested_hat_list=nested_hat,
            )

        theta = np.median(thetas)
        se = np.sqrt(np.median(np.power(ses, 2) * self.n_obs + np.power(thetas - theta, 2)) / self.n_obs)

        res = {
            "theta": theta,
            "se": se,
            "thetas": thetas,
            "ses": ses,
            "all_yx_hat": all_yx_hat,
            "all_ymx_hat": all_ymx_hat,
            "all_px_hat": all_px_hat,
            "all_pmx_hat": all_pmx_hat,
            "all_nested_hat": all_nested_hat,
        }
        return res

    def fit_nuisance_med(
        self,
        smpls,
        yx_params=None,
        ymx_params=None,
        px_params=None,
        pmx_params=None,
        nested_params=None,
    ):
        xm = np.column_stack((self.x, self.m))

        ml_yx = clone(self.learner_yx)

        if is_classifier(ml_yx):
            yx_hat_list = fit_predict_proba(y=self.y, x=self.x, ml_model=ml_yx, params=yx_params, smpls=smpls)
        else:
            yx_hat_list = fit_predict(y=self.y, x=self.x, ml_model=ml_yx, params=yx_params, smpls=smpls)

        ml_pmx = clone(self.learner_pmx)
        pmx_hat_list = fit_predict_proba(y=self.treated, x=xm, ml_model=ml_pmx, params=pmx_params, smpls=smpls)

        ml_px = clone(self.learner_px)
        # Debugging: set trimming_threshold back to the parameter
        px_hat_list = fit_predict_proba(
            y=self.treated, x=self.x, ml_model=ml_px, params=px_params, smpls=smpls, trimming_threshold=0
        )

        # ymx and nested
        mu_test_smpls = np.full([len(smpls), 2], np.ndarray)
        mu_delta_smpls = copy.deepcopy(mu_test_smpls)
        delta_test_smpls = copy.deepcopy(mu_test_smpls)

        for idx, (train, test) in enumerate(smpls):
            mu, delta = train_test_split(train, test_size=0.5)
            mu_test_smpls[idx][0] = mu
            mu_test_smpls[idx][1] = test
            mu_delta_smpls[idx][0] = mu
            mu_delta_smpls[idx][1] = delta
            delta_test_smpls[idx][0] = delta
            delta_test_smpls[idx][1] = test

        train_cond_d1 = np.where(self.treated == 1)[0]
        ml_ymx = clone(self.learner_ymx)
        ymx_hat_list = fit_predict(
            y=self.y, x=self.x, ml_model=ml_ymx, params=ymx_params, smpls=mu_test_smpls, train_cond=train_cond_d1
        )

        ymx_delta_hat_list = np.full(shape=self.n_obs, fill_value=np.nan)

        ml_ymx_delta = clone(self.learner_ymx)
        ymx_delta_preds = fit_predict(
            y=self.y, x=self.x, ml_model=ml_ymx_delta, params=ymx_params, smpls=mu_delta_smpls, train_cond=train_cond_d1
        )
        # Reorder the predictions so that we can fit and predict the nested learner.
        ymx_delta_hat_list[mu_delta_smpls[0][1]] = ymx_delta_preds[0]
        ymx_delta_hat_list[mu_delta_smpls[1][1]] = ymx_delta_preds[1]

        # TODO: Have to reassemble ymx_delta_hat_list into a single list with the good index.
        train_cond_d0 = np.where(self.treated == 0)[0]
        ml_nested = clone(self.learner_nested)
        nested_hat_list = fit_predict(
            y=ymx_delta_hat_list,
            x=xm,
            ml_model=ml_nested,
            params=nested_params,
            smpls=delta_test_smpls,
            train_cond=train_cond_d0,
        )

        return yx_hat_list, ymx_hat_list, px_hat_list, pmx_hat_list, nested_hat_list

    def compute_residuals(
        self,
        smpls,
        yx_hat_list,
        ymx_hat_list,
        px_hat_list,
        pmx_hat_list,
        nested_hat_list,
    ):

        yx_hat = np.full_like(self.y, np.nan, dtype="float64")
        ymx_hat = np.full_like(self.y, np.nan, dtype="float64")
        px_hat = np.full_like(self.y, np.nan, dtype="float64")
        pmx_hat = np.full_like(self.y, np.nan, dtype="float64")
        nested_hat = np.full_like(self.y, np.nan, dtype="float64")
        u_hat = np.full_like(self.y, np.nan, dtype="float64")
        w_hat = np.full_like(self.y, np.nan, dtype="float64")

        for idx, (_, test_index) in enumerate(smpls):
            yx_hat[test_index] = yx_hat_list[idx]
            ymx_hat[test_index] = ymx_hat_list[idx]
            px_hat[test_index] = px_hat_list[idx]
            pmx_hat[test_index] = pmx_hat_list[idx]
            nested_hat[test_index] = nested_hat_list[idx]
            u_hat[test_index] = self.y[test_index] - ymx_hat_list[idx]
            w_hat[test_index] = ymx_hat_list[idx] - nested_hat_list[idx]

        _check_is_propensity(px_hat, "learner_px", "ml_px", smpls, eps=1e-12)
        _check_is_propensity(pmx_hat, "learner_pmx", "ml_pmx", smpls, eps=1e-12)

        return yx_hat, ymx_hat, px_hat, pmx_hat, nested_hat, u_hat, w_hat

    # TODO: Probably add method to get adjusted px_hat
    def med_dml2(
        self,
        smpls,
        yx_hat_list,
        ymx_hat_list,
        px_hat_list,
        pmx_hat_list,
        nested_hat_list,
    ):

        yx_hat, ymx_hat, px_hat, pmx_hat, nested_hat, u_hat, w_hat = self.compute_residuals(
            yx_hat_list=yx_hat_list,
            ymx_hat_list=ymx_hat_list,
            px_hat_list=px_hat_list,
            pmx_hat_list=pmx_hat_list,
            nested_hat_list=nested_hat_list,
            smpls=smpls,
        )
        theta_hat = self.med_orth(
            px_hat=px_hat,
            pmx_hat=pmx_hat,
            nested_hat=nested_hat,
            u_hat=u_hat,
            w_hat=w_hat,
        )

        se = np.sqrt(
            self.var_med(
                theta=theta_hat,
                px_hat=px_hat,
                pmx_hat=pmx_hat,
                nested_hat=nested_hat,
                u_hat=u_hat,
                w_hat=w_hat,
            )
        )

        return theta_hat, se

    def med_orth(
        self,
        px_hat,
        pmx_hat,
        nested_hat,
        u_hat,
        w_hat,
    ):
        res = np.mean(
            np.multiply(np.multiply(np.divide(self.treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)), u_hat)
            + np.multiply(np.divide(1.0 - self.treated, 1.0 - px_hat), w_hat)
            + nested_hat
        )
        return res

    def var_med(self, theta, px_hat, pmx_hat, nested_hat, u_hat, w_hat):
        var = np.divide(
            np.mean(
                np.power(
                    np.multiply(np.multiply(np.divide(self.treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)), u_hat)
                    + np.multiply(np.divide(1.0 - self.treated, 1.0 - px_hat), w_hat)
                    + nested_hat
                    - theta,
                    2,
                )
            ),
            self.n_obs,
        )
        return var

    # TODO: Continue here (2025.08.07)
    def boot_med(
        self,
        thetas,
        ses,
        all_yx_hat,
        all_ymx_hat,
        all_px_hat,
        all_pmx_hat,
        all_nested_hat,
        bootstrap,
        n_rep_boot,
    ):

        all_boot_t_stat = list()
        for i_rep in range(self.n_rep):
            smpls = self.all_smpls[i_rep]

            weights = draw_weights(bootstrap, n_rep_boot, self.n_obs)
            boot_t_stat = self.boot_med_single_split(
                thetas[i_rep],
                all_yx_hat[i_rep],
                all_ymx_hat[i_rep],
                all_px_hat[i_rep],
                all_pmx_hat[i_rep],
                all_nested_hat[i_rep],
                smpls,
                ses[i_rep],
                weights,
                n_rep_boot,
            )
            all_boot_t_stat.append(boot_t_stat)

        boot_t_stat = np.hstack(all_boot_t_stat)

        return boot_t_stat

    def boot_med_single_split(
        self,
        theta,
        yx_hat_list,
        ymx_hat_list,
        px_hat_list,
        pmx_hat_list,
        nested_hat_list,
        smpls,
        se,
        weights,
        n_rep_boot,
    ):
        yx_hat, ymx_hat, px_hat, pmx_hat, nested_hat, u_hat, w_hat = self.compute_residuals(
            smpls,
            yx_hat_list,
            ymx_hat_list,
            px_hat_list,
            pmx_hat_list,
            nested_hat_list,
        )

        #    if normalize_ipw:
        #        px_hat_adj = _normalize_ipw(px_hat, treated)
        #    else:
        #        px_hat_adj = px_hat

        J = -1.0
        psi = (
            np.multiply(np.multiply(np.divide(self.treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)), u_hat)
            + np.multiply(np.divide(1.0 - self.treated, 1.0 - px_hat), w_hat)
            + nested_hat
            - theta
        )
        boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

        return boot_t_stat

    def tune_nuisance_med(
        self,
        ml_yx,
        ml_ymx,
        ml_px,
        ml_pmx,
        ml_nested,
        smpls,
        n_folds_tune,
        param_grid_g,
        param_grid_m,
    ):
        pass


# TODO: Check all non fit methods for ManualMedP
# TODO: Create manual class for efficient and efficient-alt scores.
