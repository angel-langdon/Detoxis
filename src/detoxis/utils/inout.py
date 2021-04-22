import os
import pandas as pd
import numpy as np
import datetime


DATA_PATH = './data'
RESULTS_PATH = './results'


def read_processed_data(file):
    data = pd.read_csv(os.path.join(DATA_PATH, file), encoding='utf-8')
    return data


def read_results(gs_file, preds_file):
    def process_file(file):
        with open(os.path.join(RESULTS_PATH, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        idxs, results = zip(*[line.rstrip().split('\t') for line in lines])
        return np.array(results)[np.argsort(np.array(idxs).astype(int))].astype(int)

    return process_file(gs_file), process_file(preds_file)


def write_results(labels, filename):
    with open(os.path.join(RESULTS_PATH, filename), 'w') as f:
        for i, label in enumerate(labels):
            last = ((len(labels) - 1) == i)
            f.write(f'{i}\t{label}' + '\n' * (not last))
