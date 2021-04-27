# %%
from models.baselines import *
from utils.evaluation import cross_validate
from utils.inout import *


def evaluate_baseline():
    seed = 100
    np.random.seed(seed)

    print('\nReading training data...')
    data = read_processed_data('train.csv')
    X = clean_comments(data['comment'].values)
    y = data['toxicity_level'].values

    print(f'\nPerforming cross-validation on multiple baselines...')
    # Possible baselines: RandomClassifier, BOWClassifier, ChainBOW, Word2VecSpacy, GloVeSBWC
    baselines = [SVMClassifier]
    n_folds = 2
    for baseline in baselines:
        cross_validate(X, y, n_folds, baseline, seed)


def evaluate_results():
    gs_file = 'gold_standard.txt'
    preds_file = 'predictions.txt'

    gs, preds = read_results(gs_file, preds_file)

    evaluator = Evaluator()
    print(evaluator.evaluate(gs, preds))


if __name__ == '__main__':
    evaluate_baseline()
    # evaluate_results()

# %%
