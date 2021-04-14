# %%
import os

import pandas as pd
from utils.path_utils import paths


# %%
def read_train_dataset():
    train_dataset_path = os.path.join(paths.DATASETS, "train.csv")
    df = pd.read_csv(train_dataset_path)
    return df


df = read_train_dataset()
df
# %%
