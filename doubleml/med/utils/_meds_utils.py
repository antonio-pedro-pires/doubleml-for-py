import pandas as pd


def generate_effects_summary(effects):
    df_effects = pd.DataFrame()
    for _, effect in effects.items():
        df_effects = pd.concat([df_effects, effect.summary])
    df_effects.index = list(effects)
    return df_effects
