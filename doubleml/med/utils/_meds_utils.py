
import pandas as pd


def generate_effects_summary(effects):
    rows = {
        "ATE": None,
        "DIR_TREAT": None,
        "DIR_CONTROL": None,
        "INDIR_TREAT": None,
        "INDIR_CONTROL": None,
    }
    col_names = ["coef", "std err", "t", "P>|t|"]
    for row in effects:
        rows[row] = [
            effects[row].all_thetas[0][0],
            effects[row].all_ses[0][0],
            effects[row].all_t_stats[0][0],
            effects[row].all_pvals[0][0],
        ]
    df_effects = pd.DataFrame.from_dict(rows, orient="index", columns=col_names)
    return df_effects
