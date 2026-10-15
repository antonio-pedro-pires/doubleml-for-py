"""Helper to aggregate the effect summaries of the joint mediation estimator."""

from typing import Any

import pandas as pd


def generate_effects_summary(effects: dict[str, Any]) -> pd.DataFrame:
    """
    Stack the summaries of the effects into one data frame, with one row per effect.

    Parameters
    ----------
    effects : dict[str, DoubleMLFramework]
        The frameworks of the effects, keyed by the name of the effect.

    Returns
    -------
    pd.DataFrame
        The summaries of all effects, indexed by the names of the effects.
    """
    if not effects:
        return pd.DataFrame(index=[])

    df_effects = pd.concat([effect.summary for effect in effects.values()])
    df_effects.index = list(effects)
    return df_effects
