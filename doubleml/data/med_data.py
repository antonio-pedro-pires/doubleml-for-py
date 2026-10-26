"""Data container for causal mediation analysis with mediators."""

import io
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.multiclass import type_of_target

from doubleml.data.base_data import DoubleMLBaseData, DoubleMLData
from doubleml.utils._estimation import _assure_2d_array


class DoubleMLMEDData(DoubleMLData):
    """
    Data-container for causal mediation analysis.

    :class:`DoubleMLMEDData` objects can be initialized from a
    :class:`pandas.DataFrame` via :meth:`DoubleMLMEDData.__init__` or from
    :class:`numpy.ndarray` via the class method :meth:`DoubleMLMEDData.from_arrays`.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data.
    y_col : str
        The name of the outcome variable :math:`Y`.
    d_cols : str or list
        The name of the treatment variable :math:`D`.
    m_cols : str or list
        The name of the mediator variable(s) :math:`M`.
    x_cols : str or list, optional
        The names of the covariates :math:`X`.
        Default is ``None``.
    z_cols : str or list, optional
        The names of the instrumental variables :math:`Z`.
        Default is ``None``.
    use_other_treat_as_covariate : bool, optional
        Indicates whether in the case of multiple treatment variables the other
        treatment variables should be added as covariates.
        Default is ``True``.
    force_all_x_finite : bool, optional
        Indicates whether to raise an error if there are missing or infinite
        values in the covariates :math:`X`. Possible values are: ``True`` (neither missings
        ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed), ``False`` (missings and
        infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        Default is ``True``.
    force_all_d_finite : bool, optional
        Indicates whether to raise an error if there are missing or infinite
        values in the treatment variable :math:`D`. Possible values are: ``True`` (neither missings
        ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed), ``False`` (missings and
        infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        Default is ``True``.
    force_all_m_finite : bool, optional
        Indicates whether to raise an error if there are missing or infinite
        values in the mediators :math:`M`. Possible values are: ``True`` (neither missings
        ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed), ``False`` (missings and
        infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
        Default is ``True``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from doubleml import DoubleMLMEDData
    >>> df = pd.DataFrame({
    ...     'y': np.random.randn(100),
    ...     'd': np.random.randint(0, 2, 100),
    ...     'm1': np.random.randn(100),
    ...     'm2': np.random.randint(0, 2, 100),
    ...     'x1': np.random.randn(100),
    ...     'x2': np.random.randn(100)
    ... })
    >>> med_data = DoubleMLMEDData(data=df,
    ...                            y_col='y',
    ...                            d_cols='d',
    ...                            m_cols=['m1', 'm2'],
    ...                            x_cols=['x1', 'x2'],
    ...                            force_all_m_finite=True)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y_col: str,
        d_cols: Union[str, list[str]],
        m_cols: Union[str, list[str]],
        x_cols: Optional[Union[str, list[str]]] = None,
        z_cols: Optional[Union[str, list[str]]] = None,
        use_other_treat_as_covariate: bool = True,
        force_all_x_finite: Union[bool, str] = True,
        force_all_d_finite: Union[bool, str] = True,
        force_all_m_finite: Union[bool, str] = True,
    ) -> None:

        DoubleMLBaseData.__init__(self, data)

        # we need to set m (needs _data) before call to the super __init__ because of the x_cols setter.
        self.m_cols = m_cols

        # Declare the type of _force_all_m_finite without a value, so the attribute does not exist yet and the setter
        # only validates and assigns. It has to be set through the setter before _set_m().
        self._force_all_m_finite: Union[bool, str]
        self.force_all_m_finite = force_all_m_finite
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

    def __str__(self) -> str:
        data_summary = self._data_summary_str()
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        df_info = buf.getvalue()
        res = (
            f"================== {self.__class__.__name__} Object ==================\n"
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ DataFrame info    ------------------\n"
            + df_info
        )
        return res

    def _data_summary_str(self) -> str:
        data_summary = (
            f"Outcome variable: {self.y_col}\n"
            f"Treatment variable(s): {self.d_cols}\n"
            f"Mediation variable(s): {self.m_cols}\n"
            f"Covariates: {self.x_cols}\n"
            f"Instrument variable(s): {self.z_cols}\n"
            f"No. Observations: {self.n_obs}\n"
        )
        return data_summary

    @classmethod
    def from_arrays(  # type: ignore[override]  # med adds the mediators m, so the signature differs from the base class
        cls,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        m: np.ndarray,
        z: Optional[np.ndarray] = None,
        use_other_treat_as_covariate: bool = False,
        force_all_x_finite: Union[bool, str] = True,
        force_all_d_finite: Union[bool, str] = True,
        force_all_m_finite: Union[bool, str] = True,
    ) -> "DoubleMLMEDData":
        """
        Initialize :class:`DoubleMLMEDData` from :class:`numpy.ndarray` objects.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The covariates :math:`X`.
        y : :class:`numpy.ndarray`
            The outcome variable :math:`Y`.
        d : :class:`numpy.ndarray`
            The treatment variable :math:`D`.
        m : :class:`numpy.ndarray`
            The mediator variable(s) :math:`M`.
        z : :class:`numpy.ndarray`, optional
            The instrumental variables :math:`Z`.
            Default is ``None``.
        use_other_treat_as_covariate : bool, optional
            Indicates whether in the case of multiple treatment variables the
            other treatment variables should be added as covariates.
            Default is ``False``.
        force_all_x_finite : bool, optional
            Indicates whether to raise an error if there are missing or
            infinite values in the covariates :math:`X`. Possible values are: ``True`` (neither
            missings ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed), ``False``
            (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
            Default is ``True``.
        force_all_d_finite : bool, optional
            Indicates whether to raise an error if there are missing or
            infinite values in the treatment variable :math:`D`. Possible values are: ``True``
            (neither missings ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed),
            ``False`` (missings and infinite values are allowed), ``'allow-nan'`` (only missings
            are allowed). Default is ``True``.
        force_all_m_finite : bool, optional
            Indicates whether to raise an error if there are missing or
            infinite values in the mediators :math:`M`. Possible values are: ``True`` (neither
            missings ``np.nan``/``pd.NA`` nor infinite values ``np.inf`` are allowed), ``False``
            (missings and infinite values are allowed), ``'allow-nan'`` (only missings are allowed).
            Default is ``True``.

        Returns
        -------
        obj : :class:`DoubleMLMEDData`
            The initialized :class:`DoubleMLMEDData` object.

        Example
        -------

        >>> import numpy as np
        >>> from doubleml import DoubleMLMEDData
        >>> # Create numpy arrays
        >>> x = np.random.randn(100, 2)
        >>> y = np.random.randn(100)
        >>> d = np.random.randint(0, 2, 100)
        >>> m = np.random.randn(100, 1)
        >>> # Initialize from arrays
        >>> med_data_arrays = DoubleMLMEDData.from_arrays(x,
        ...                                               y,
        ...                                               d,
        ...                                               m,
        ...                                               force_all_x_finite=True,
        ...                                               force_all_m_finite=True)
        """

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

        dml_data = DoubleMLData.from_arrays(
            x=x,
            y=y,
            d=d,
            z=z,
            cluster_vars=None,
            use_other_treat_as_covariate=use_other_treat_as_covariate,
            force_all_x_finite=force_all_x_finite,
            force_all_d_finite=force_all_d_finite,
        )
        m = check_array(m, ensure_2d=False, allow_nd=False, ensure_all_finite=force_all_m_finite, input_name="m")
        m = _assure_2d_array(m)

        if m.shape[1] == 1:
            m_cols = ["m"]
        else:
            m_cols = [f"m{i + 1}" for i in np.arange(m.shape[1])]

        data = pd.concat([dml_data.data, (pd.DataFrame(m, columns=m_cols))], axis=1)

        return cls(
            data=data,
            y_col=dml_data.y_col,
            d_cols=dml_data.d_cols,
            m_cols=m_cols,
            x_cols=dml_data.x_cols,
            z_cols=dml_data.z_cols,
            use_other_treat_as_covariate=dml_data.use_other_treat_as_covariate,
            force_all_x_finite=dml_data.force_all_x_finite,
            force_all_d_finite=dml_data.force_all_d_finite,
            force_all_m_finite=force_all_m_finite,
        )

    @property
    def m(self) -> Optional[np.ndarray]:
        """
        Array of mediation variables.
        """
        if self.m_cols is not None:
            return self._m.values
        else:
            return None

    @property
    def m_cols(self) -> list[str]:
        """
        The mediation variable(s).
        """
        return self._m_cols

    @m_cols.setter
    def m_cols(self, value: Union[str, list[str]]) -> None:
        reset_value = hasattr(self, "_m_cols")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                "The mediation variable(s) m_cols must be of str or list type. "
                f"{str(value)} of type {str(type(value))} was passed."
            )
        if not len(set(value)) == len(value):
            raise ValueError("Invalid mediation variable(s) m_cols: Contains duplicate values.")
        if len(value) == 0:
            raise ValueError("Invalid mediation variable(s) m_cols: At least one mediation variable is required.")
        if not set(value).issubset(set(self.all_variables)):
            raise ValueError("Invalid mediation variable(s) m_cols. At least one mediation variable is no data column.")
        if reset_value:
            # validate the candidate first, assign only after every check passed
            super()._check_disjoint_sets()
            self._check_disjoint_sets_m_cols(value)
            new_m = self._get_m(value, self._force_all_m_finite)
            new_binary_meds = self._check_binary_mediators(value, self._force_all_m_finite)
            self._m = new_m
            self._binary_meds = new_binary_meds
        self._m_cols = value

    @property
    def cluster_cols(self) -> Optional[list[str]]:
        """
        The cluster variable(s). Always ``None``, cluster data is not supported for mediation analysis.
        """
        return self._cluster_cols

    @cluster_cols.setter
    def cluster_cols(self, value: Optional[Union[str, list[str]]]) -> None:
        if value is not None:
            raise NotImplementedError(
                f"Cluster variables are not supported for mediation analysis. Expected cluster_cols=None, got {value!r}."
            )
        getattr(DoubleMLData, "cluster_cols").fset(self, value)

    @property
    def force_all_m_finite(self) -> Any:
        """
        Indicates whether to raise an error on infinite values and / or missings in the mediators ``m``.
        """
        return self._force_all_m_finite

    @force_all_m_finite.setter
    def force_all_m_finite(self, value: Union[bool, str]) -> None:
        reset_value = hasattr(self, "_force_all_m_finite")
        if isinstance(value, str):
            if value != "allow-nan":
                raise ValueError(
                    f"Invalid force_all_m_finite: {str(value)}. force_all_m_finite must be True, False or 'allow-nan'."
                )
        elif not isinstance(value, bool):
            raise TypeError("Invalid force_all_m_finite. force_all_m_finite must be True, False or 'allow-nan'.")
        if reset_value:
            # validate the candidate first, assign only after every check passed
            new_m = self._get_m(self.m_cols, value)
            new_binary_meds = self._check_binary_mediators(self.m_cols, value)
            self._m = new_m
            self._binary_meds = new_binary_meds
        self._force_all_m_finite = value

    @property
    def n_meds(self) -> int:
        """
        The number of mediator variables
        """
        return len(self.m_cols)

    @property
    def binary_meds(self) -> pd.Series:
        """
        Series with logical(s) indicating whether the mediator variable(s) are binary with values 0 and 1.
        """
        return self._binary_meds

    def _get_optional_col_sets(self) -> list[set[str]]:
        base_optional_col_sets = super()._get_optional_col_sets()
        m_cols_set = set(self.m_cols)
        return [m_cols_set] + base_optional_col_sets

    def _check_binary_mediators(
        self, m_cols: Optional[list[str]] = None, force_all_m_finite: Optional[Union[bool, str]] = None
    ) -> pd.Series:
        # candidate values let the setters validate before assigning; None falls back to the current attributes
        if m_cols is None:
            m_cols = self.m_cols
        if force_all_m_finite is None:
            force_all_m_finite = self._force_all_m_finite
        is_binary = pd.Series(dtype=bool, index=m_cols)
        if force_all_m_finite is False:
            # missing and infinite values may be present, so the mediators are not checked
            is_binary[:] = False
        else:
            for m_var in m_cols:
                # for force_all_m_finite='allow-nan' only the observed values are checked, since type_of_target() rejects NaN
                this_m = self.data.loc[:, m_var].dropna()
                binary_m = this_m.size > 0 and type_of_target(this_m) == "binary"
                zero_one_treat = np.all((np.power(this_m, 2) - this_m) == 0)
                is_binary[m_var] = bool(binary_m and zero_one_treat)
        return is_binary

    def _check_disjoint_sets(self) -> None:
        super()._check_disjoint_sets()
        self._check_disjoint_sets_m_cols()

    def _check_disjoint_sets_m_cols(self, m_cols: Optional[list[str]] = None) -> None:
        # Disjointedness check for mediator variables; a candidate m_cols is checked before it is assigned.
        if m_cols is None:
            m_cols = self.m_cols
        m_cols_set = set(m_cols)
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

    def _get_m(self, m_cols: list[str], force_all_m_finite: Union[bool, str]) -> pd.DataFrame:
        # validates the mediators for the given candidate values without assigning anything
        m = self.data.loc[:, m_cols]
        if force_all_m_finite:
            assert_all_finite(m, allow_nan=force_all_m_finite == "allow-nan")
        return m

    def _set_m(self) -> None:
        self._m = self._get_m(self.m_cols, self._force_all_m_finite)
