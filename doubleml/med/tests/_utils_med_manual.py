import numpy as np
from sklearn.base import clone, is_classifier

from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search
from ...tests._utils_boot import boot_manual, draw_weights
from ...utils._checks import _check_is_propensity
from ...utils._propensity_score import _normalize_ipw


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
        m_params=None,
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
                m_params=m_params,
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
        m_params=None,
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
        m_hat_list = fit_predict_proba(self.treated, self.x, ml_m, m_params, smpls, trimming_threshold=self.trimming_threshold)
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
            g_hat1 = predictions["ml_g_d1"][:, i_rep, 0]

            weights = np.ones_like(d)
            weights_bar = np.ones_like(d)

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
        m_params=None,
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
                m_params=m_params,
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
        m_params=None,
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
        m_hat_list = fit_predict_proba(treated, x, ml_m, m_params, smpls, trimming_threshold=0)

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


class ManualMedCAlt:
    def fit_med(
        y,
        x,
        d,
        m,
        learner_g,
        learner_m,
        learner_nested,
        treatment_level,
        all_smpls,
        n_rep=1,
        g_d1_params=None,
        g_nested_params=None,
        m_med_params=None,
        m_params=None,
        normalize_ipw=False,
        trimming_threshold=1e-2,
    ):
        n_obs = len(y)
        treated = d == treatment_level
        mediated = m == treatment_level

        thetas = np.zeros(n_rep)
        ses = np.zeros(n_rep)

        # "ml_g_d1", "ml_g_nested", "ml_m", "ml_m_med"
        all_g_d1_hat = list()
        all_g_nested_hat = list()
        all_m_med_hat = list()
        all_m_hat = list()

        for i_rep in range(n_rep):
            smpls = all_smpls[i_rep]
            g_d1_hat, g_nested_hat, m_med_hat, m_hat = ManualMedCAlt.fit_nuisance_med(
                y=y,
                x=x,
                d=d,
                m=m,
                treated=treated,
                mediated=mediated,
                learner_g=learner_g,
                learner_m=learner_m,
                learner_nested=learner_nested,
                smpls=smpls,
                g_d1_params=g_d1_params,
                g_nested_params=g_nested_params,
                m_med_params=m_med_params,
                m_params=m_params,
                trimming_threshold=trimming_threshold,
            )

            all_g_d1_hat.append(g_d1_hat)
            all_g_nested_hat.append(g_nested_hat)
            all_m_med_hat.append(m_med_hat)
            all_m_hat.append(m_hat)

            thetas[i_rep], ses[i_rep] = ManualMedCAlt.med_dml_2(
                y,
                x,
                d,
                m,
                treated,
                mediated,
                g_d1_hat,
                g_nested_hat,
                m_med_hat,
                m_hat,
                smpls,
                normalize_ipw,
            )

        theta = np.median(thetas)
        se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

        res = {
            "theta": theta,
            "se": se,
            "thetas": thetas,
            "ses": ses,
            "all_g_d1_hat": g_d1_hat,
            "all_g_nested_hat": g_nested_hat,
            "all_m_med_hat": m_med_hat,
            "all_m_hat": all_m_hat,
        }
        return res

    def fit_nuisance_med(
        y,
        x,
        m,
        treated,
        learner_g,
        learner_m,
        learner_med,
        learner_nested,
        smpls,
        med_params=None,
        g_d1_params=None,
        g_nested_params=None,
        m_params=None,
        trimming_threshold=1e-12,
    ):
        xm = np.column_stack((x, m))

        # TODO: Separate samples into delta, musample, testsample
        # g_d1_hat, g_nested_hat, m_med_hat, m_hat
        ml_g_d1 = clone(learner_g)
        ml_g_nested = clone(learner_nested)
        # TODO: What learner is ml_m_med? Does it reuse the m_learner?
        ml_m_med = clone(learner_med)

        # TODO: Find way to select the intersection between m and d for train_cond_xy,
        # where x, y in [0, 1] and xy = cartesian_product(x, y)
        # For 1-dimensional d and m.

        train_cond_d1 = np.where(treated == 1)[0]
        if is_classifier(learner_g):
            g_d1_hat_list = fit_predict_proba(y, xm, ml_g_d1, g_d1_params, smpls, train_cond=train_cond_d1)
        else:
            g_d1_hat_list = fit_predict(y, xm, ml_g_d1, g_d1_params, smpls, train_cond=train_cond_d1)

        train_cond_d0 = np.where(treated == 0)[0]
        if is_classifier(ml_g_nested):
            g_nested_hat_list = fit_predict_proba(g_d1_hat_list, x, g_nested_params, smpls, train_cond=train_cond_d0)
        else:
            g_nested_hat_list = fit_predict(g_d1_hat_list, x, g_nested_params, smpls, train_cond=train_cond_d0)

        # TODO: shouldn't learner_med be a classifier??
        if is_classifier(learner_med):
            # TODO: Apply trimming-threshold???
            m_med_hat_list = fit_predict_proba(y, x, ml_m_med, med_params, smpls)
        else:
            m_med_hat_list = fit_predict(y, x, ml_m_med, med_params, smpls)

        ml_m = clone(learner_m)
        m_hat_list = fit_predict_proba(treated, x, ml_m, m_params, smpls, trimming_threshold=trimming_threshold)

        # g_d1_hat, g_nested_hat, m_med_hat, m_hat
        return g_d1_hat_list, g_nested_hat_list, m_med_hat_list, m_hat_list

    def compute_residuals(
        y,
        smpls,
        m_hat_list,
        g_d1_m0_hat_list,
        g_d1_hat_list,
        g_nested_hat_list,
        m_med_hat_list,
    ):

        g_d1_hat = np.full_like(y, np.nan, dtype="float64")
        g_nested_hat = np.full_like(y, np.nan, dtype="float64")
        m_med_hat = np.full_like(y, np.nan, dtype="float64")
        m_hat = np.full_like(y, np.nan, dtype="float64")
        u_hat = np.full_like(y, np.nan, dtype="float64")
        w_hat = np.full_like(y, np.nan, dtype="float64")

        for idx, (_, test_index) in enumerate(smpls):
            g_d1_hat[test_index] = g_d1_m0_hat_list[idx]
            g_nested_hat[test_index] = g_nested_hat_list[idx]
            m_med_hat[test_index] = m_med_hat_list[idx]
            m_hat[test_index] = m_hat_list[idx]
            u_hat[test_index] = y[test_index] - g_d1_hat_list[idx]
            w_hat[test_index] = g_d1_hat[idx] - g_nested_hat_list[idx]

        _check_is_propensity(m_hat, "learner_m", "ml_m", smpls, eps=1e-12)
        return g_d1_hat, g_nested_hat, m_med_hat, m_hat, u_hat, w_hat

    # TODO: Probably add method to get adjusted m_hat
    def med_dml2(
        y,
        m,
        treated,
        smpls,
        normalize_ipw,
        m_hat_list,
        g_d1_hat_list,
        g_nested_hat_list,
        m_med_hat_list,
    ):
        n_obs = len(y)

        g_d1_hat, g_nested_hat, m_med_hat, m_hat, u_hat, w_hat = ManualMedCAlt.compute_residuals(
            y=y,
            m=m,
            g_d1_hat_list=g_d1_hat_list,
            g_nested_hat_list=g_nested_hat_list,
            m_med_hat_list=m_med_hat_list,
            m_hat_list=m_hat_list,
            smpls=smpls,
        )
        theta_hat = ManualMedCAlt.med_orth(
            treated=treated,
            u_hat=u_hat,
            w_hat=w_hat,
            g_nested_hat=g_nested_hat,
            m_med_hat=m_med_hat,
            m_hat=m_hat,
        )

        se = np.sqrt(
            ManualMedCAlt.var_med(
                theta_hat=theta_hat,
                g_d1_hat=g_d1_hat,
                g_nested_hat=g_nested_hat,
                m_med_hat=m_med_hat,
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
        g_nested_hat=None,
        m_med_hat=None,
        m_hat=None,
    ):
        res = np.mean(
            np.multiply(np.multiply(np.divide(treated, 1.0 - m_hat), np.divide(1.0 - m_med_hat, m_med_hat)), u_hat)
            + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
            + g_nested_hat
        )
        return res

    def var_med(theta, treated, n_obs, m_hat, u_hat, w_hat, m_med_hat, g_nested_hat):
        var = (
            1
            / n_obs
            * np.mean(
                np.power(
                    np.multiply(np.multiply(np.divide(treated, 1.0 - m_hat), np.divide(1.0 - m_med_hat, m_med_hat)), u_hat)
                    + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
                    + g_nested_hat
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
        all_g_d1_hat,
        all_g_nested_hat,
        all_m_med_hat,
        all_m_hat,
        bootstrap,
        n_rep_boot,
        all_smpls,
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
            boot_t_stat = ManualMedCAlt.boot_med_single_split(
                thetas[i_rep],
                y,
                d,
                m,
                treated,
                mediated,
                all_g_d1_hat[i_rep],
                all_g_nested_hat[i_rep],
                all_m_med_hat[i_rep],
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
        d,
        m,
        treated,
        m_hat_list,
        g_d1_m0_hat_list,
        g_d1_hat_list,
        g_nested_hat_list,
        m_med_hat_list,
        smpls,
        se,
        weights,
        n_rep_boot,
        normalize_ipw,
    ):
        g_d1_hat, g_nested_hat, m_med_hat, m_hat, u_hat, w_hat = ManualMedCAlt.compute_residuals(
            y,
            m,
            smpls,
            m_hat_list,
            g_d1_m0_hat_list,
            g_d1_hat_list,
            g_nested_hat_list,
            m_med_hat_list,
        )

        #    if normalize_ipw:
        #        m_hat_adj = _normalize_ipw(m_hat, treated)
        #    else:
        #        m_hat_adj = m_hat

        J = -1.0
        psi = (
            np.multiply(np.multiply(np.divide(treated, 1.0 - m_hat), np.divide(1.0 - m_med_hat, m_med_hat)), u_hat)
            + np.multiply(np.divide(1.0 - treated, 1.0 - m_hat), w_hat)
            + g_nested_hat
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


# TODO: Check all non fit methods for ManualMedP
# TODO: Create manual class for efficient and efficient-alt scores.
