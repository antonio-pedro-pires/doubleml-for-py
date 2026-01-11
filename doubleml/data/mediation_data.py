import io

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.multiclass import type_of_target

from doubleml.data.base_data import DoubleMLBaseData, DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLMediationData(DoubleMLData):

    # TODO: Check if we can use other treatments as covariates during mediation analysis
    #  and if the same is true for mediators.
    def __init__(
        self,
        data,
        y_col,
        d_cols,
        m_cols,
        x_cols=None,
        z_cols=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
        force_all_m_finite=True,
    ):

        DoubleMLBaseData.__init__(self, data)

        # we need to set m (needs _data) before call to the super __init__ because of the x_cols setter.
        self.m_cols = m_cols

        # Need to set _force_all_m_finite before _set_m()
        self._force_all_m_finite = force_all_m_finite
        self._set_m()

        DoubleMLData.__init__(
            self,
            data,
            y_col,
            d_cols,
            x_cols,
            z_cols,
            cluster_cols=None,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=force_all_d_finite,
        )

        self._check_disjoint_sets_m_cols()
        self._binary_meds = self._check_binary_mediators()

    def __str__(self):
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            "================== DoubleMLMediationData Object ==================\n"
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ DataFrame info    ------------------\n"
            + df_info
        )
        return res

    def _data_summary_str(self):
        data_summary = (
            f"Outcome variable: {self.y_col}\n"
            f"Treatment variable(s): {self.d_cols}\n"
            f"Mediation variable(s): {self.m_cols}\n"
            f"Covariates: {self.x_cols}\n"
            f"Instrument variable(s): {self.z_cols}\n"
        )

        data_summary += f"No. Observations: {self.n_obs}\n"
        return data_summary

    @classmethod
    def from_arrays(
        cls,
        x,
        y,
        d,
        m,
        z=None,
        use_other_treat_as_covariate=True,
        force_all_x_finite=True,
        force_all_d_finite=True,
        force_all_m_finite=True,
    ):

        if isinstance(force_all_m_finite, str):
            if force_all_m_finite != "allow-nan":
                raise ValueError(
                    "Invalid force_all_m_finite "
                    + force_all_m_finite
                    + ". "
                    + "force_all_m_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(force_all_m_finite, bool):
            raise TypeError(
                "Invalid force_all_m_finite. "
                + "force_all_m_finite must be True, False or 'allow-nan'. "
                + f"{str(force_all_m_finite)} of type {str(type(force_all_m_finite))} was passed."
            )

        dml_data = DoubleMLData.from_arrays(x, y, d, z, None, use_other_treat_as_covariate, force_all_x_finite)
        m = check_array(m, ensure_2d=False, allow_nd=False, force_all_finite=force_all_m_finite)
        m = _assure_2d_array(m)

        if m.shape[1] == 1:
            m_cols = ["m"]
        else:
            m_cols = [f"m{i + 1}" for i in np.arange(m.shape[1])]

        data = pd.concat([dml_data.data, (pd.DataFrame(m, columns=m_cols))], axis=1)

        return cls(
            data,
            dml_data.y_col,
            dml_data.d_cols,
            m_cols,
            dml_data.x_cols,
            dml_data.z_cols,
            dml_data.t_col,
            dml_data.s_col,
            dml_data.use_other_treat_as_covariate,
            dml_data.force_all_x_finite,
        )

    @property
    def m(self):
        """
        Array of mediation variables.
        """
        if self.m_cols is not None:
            return self._m.values
        else:
            return None

    @property
    def m_cols(self):
        """
        The mediation variable(s).
        """
        return self._m_cols

    @m_cols.setter
    def m_cols(self, value):
        reset_value = hasattr(self, "_m_cols")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                "The mediation variable(s) m_cols must be of str or list type (or None). "
                f"{str(value)} of type {str(type(value))} was passed."
            )
        if not len(set(value)) == len(value):
            raise ValueError("Invalid mediation variable(s) m_cols: Contains duplicate values.")
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError("Invalid mediation variable(s) m_cols. At least one mediation variable is no data column.")
        self._m_cols = value
        if reset_value:
            self._check_disjoint_sets()
            self._set_m()

    @property
    def force_all_m_finite(self):
        """
        Indicates whether to raise an error on infinite values and / or missings in the mediators ``m``.
        """
        return self._force_all_m_finite

    @force_all_m_finite.setter
    def force_all_m_finite(self, value):
        reset_value = hasattr(self, "_force_all_m_finite")
        if isinstance(value, str):
            if value != "allow-nan":
                raise ValueError(
                    "Invalid force_all_m_finite " + value + ". " + "force_all_m_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(value, bool):
            raise TypeError("Invalid force_all_m_finite. " + "force_all_m_finite must be True, False or 'allow-nan'.")
        self._force_all_m_finite = value
        if reset_value:
            self._set_m()

    @property
    def n_meds(self):
        """
        The number of mediator variables
        """
        return len(self.m_cols)

    @property
    def binary_meds(self):
        """
        Series with logical(s) indicating whether the mediator variable(s) are binary with values 0 and 1.
        """
        return self._binary_meds

    def _get_optional_col_sets(self):
        base_optional_col_sets = super()._get_optional_col_sets()
        m_cols_set = set(self.m_cols)
        return [m_cols_set] + base_optional_col_sets

    def _check_binary_mediators(self):
        is_binary = pd.Series(dtype=bool, index=self.m_cols)
        if not self._force_all_m_finite:
            is_binary[:] = False  # if we allow infinite values, we cannot check for binary
        else:
            for m_var in self.m_cols:
                this_m = self.data.loc[:, m_var]
                binary_m = type_of_target(this_m) == "binary"
                zero_one_treat = np.all((np.power(this_m, 2) - this_m) == 0)
                is_binary[m_var] = binary_m & zero_one_treat
        return is_binary

    def _check_disjoint_sets(self):
        super()._check_disjoint_sets()
        self._check_disjoint_sets_m_cols()

    def _check_disjoint_sets_m_cols(self):
        # TODO: Is this truly necessary, since the _check_disjoint_sets()
        #  already calls its super which calls for checks it itself?

        # apply the standard checks from the DoubleMLData class
        super(DoubleMLMediationData, self)._check_disjoint_sets()

        # Disjointedness check for mediator variables.
        m_cols_set = set(self.m_cols)
        y_col_set = {self.y_col}
        x_cols_set = set(self.x_cols)
        d_cols_set = set(self.d_cols)
        z_cols_set = set(self.z_cols or [])

        mediator_checks_args = [
            (y_col_set, "outcome variable", "``y_col``"),
            (d_cols_set, "treatment variable", "``d_cols``"),
            (x_cols_set, "covariate", "``x_cols``"),
            (z_cols_set, "instrumental variable", "``z_cols``"),
        ]

        for set1, name, argument in mediator_checks_args:
            self._check_disjoint(
                set1=set1,
                name1=name,
                arg1=argument,
                set2=m_cols_set,
                name2="mediation variable(s)",
                arg2="``m_cols``",
            )

    def _set_m(self):
        if self._force_all_m_finite:
            assert_all_finite(self.data.loc[:, self.m_cols], allow_nan=self._force_all_m_finite == "allow-nan")
        self._m = self.data.loc[:, self.m_cols]
