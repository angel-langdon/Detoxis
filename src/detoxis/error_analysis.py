# %%
import pandas as pd

toxic_path = "./data/error_toxic.xlsx"
toxicity_level_path = "./data/error_toxicicty_level.xlsx"


def separate_bad_labeled_rows(path):
    df = pd.read_excel(path)
    error_cols = [col for col in df.columns if "error" in col]
    bad_dfs = []
    for c in error_cols:
        real_col = [c for c in df.columns if "real" in c][0]
        prediction_col = [c for c in df.columns if "prediction" in c][0]
        bool_indexing = ((~df[c].isin([0, 1])) &
                         (df[real_col] != df[prediction_col]))
        bad_formatted_rows = df[bool_indexing]
        bad_dfs.append(bad_formatted_rows)
        df = df.drop(df[bool_indexing].index)
    pd.concat(bad_dfs).to_excel("bad_rows.xlsx", index=False)
    df.to_excel("correct_rows.xlsx", index=False)


def preprocess_errors_excel(path):
    df = pd.read_excel(path)
    real_col = [c for c in df.columns if "real" in c][0]
    prediction_col = [c for c in df.columns if "prediction" in c][0]
    df = df[df[real_col] != df[prediction_col]]
    return df


def get_error_percentages(df: pd.DataFrame, task):
    error_cols = [col for col in df.columns if "error" in col]
    error_dic = {col.replace("error_", ""): f"{round(100*df[col].sum()/len(df), 2)} %"
                 for col in error_cols}
    error_dic["task"] = task
    return error_dic


errors_rows = []
df = preprocess_errors_excel(toxic_path)
error_percentages = get_error_percentages(df, "toxicity")
errors_rows.append(error_percentages)

# %%
df = preprocess_errors_excel(toxicity_level_path)
error_percentages = get_error_percentages(df, "toxicity_level")
errors_rows.append(error_percentages)

# %%
pd.DataFrame(errors_rows).to_excel("./data/errors_analysis_summary.xlsx",
                                   index=False)
# %%
