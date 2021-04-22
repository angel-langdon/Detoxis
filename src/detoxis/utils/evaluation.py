import numpy as np

from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold


def cross_validate(X, y, n_folds, model_class, random_seed=None, show_progress=False):
    model_name = model_class().__class__.__name__
    print('\n' + '#' * int(len(model_name) * 2))
    print(' ' * int(len(model_name) / 2) + model_name)
    print('#' * int(len(model_name) * 2) + '\n' * show_progress)

    kfold = StratifiedKFold(n_folds, shuffle=True, random_state=random_seed)
    fold = 1
    all_results = {'f1_toxic': [],
                   'cem': [],
                   'rbp': [],
                   'pearson': []}
    for trn_idxs, tst_idxs in kfold.split(X, y):
        if show_progress:
            print(f'+++ Processing fold {fold} / {n_folds}...')
        X_train, X_test = X[trn_idxs], X[tst_idxs]
        y_train, y_test = y[trn_idxs], y[tst_idxs]

        # Initialize and train the baseline model
        model = model_class()
        model.fit(X_train, y_train)

        # Evaluate the trained model
        results = model.score(X_test, y_test)
        fold += 1

        for key, val in results.items():
            all_results[key].append(val)

    print(f'\nRESULTS:\n\t'
          f'F1-Toxic: {np.mean(all_results["f1_toxic"]):.2f} \u00b1 {np.std(all_results["f1_toxic"]):.2f}\n\t'
          f'CEM: {np.mean(all_results["cem"]):.2f} \u00b1 {np.std(all_results["cem"]):.2f}\n\t'
          f'RBP: {np.mean(all_results["rbp"]):.2f} \u00b1 {np.std(all_results["rbp"]):.2f}\n\t'
          f'Pearson Coefficient: {np.mean(all_results["pearson"]):.2f} \u00b1 {np.std(all_results["pearson"]):.2f}')

    return all_results


class Evaluator:
    @staticmethod
    def _f1_toxic(y, y_pred, labels=(0, 1)):
        return f1_score(y, y_pred, labels=labels, average=None)[1]

    def _cem_score(self, y, y_pred):
        def _proximity(g_d, s_d, n_c):
            if g_d > s_d:
                return - np.log2((n_c[s_d] / 2 + np.sum(n_c[s_d + 1:g_d + 1])) / np.sum(n_c))
            elif g_d < s_d:
                return - np.log2((n_c[s_d] / 2 + np.sum(n_c[g_d:s_d])) / np.sum(n_c))
            else:
                return - np.log2(n_c[s_d] / 2 / np.sum(n_c))

        labels = [0, 1, 2, 3]
        gs_cs = [list(y).count(label) for label in labels]

        num = [_proximity(g, s, gs_cs) for g, s in zip(y, y_pred)]
        den = [_proximity(g, g, gs_cs) for g in y]
        return sum(num) / sum(den)

    @staticmethod
    def _rbp(y, y_pred):
        def get_relevance(label):
            # Relevance defined as the toxicity level normalized by the maximum relevance
            return (2 ** label - 1) / 2 ** labels[-1]

        def get_estimated_prob(label, counts):
            # Get the number of cases above the predicted class in a ranking (assuming an ordered list of label counts)
            offset = np.sum((counts + [0])[label + 1:])
            # Get all possible positions in a ranking for a possible label
            idxs = np.arange(offset, offset + counts[label])
            # If the label appears at least once in the predictions, compute the estimated probability of that class
            if len(idxs) > 0:
                return np.mean(p ** idxs)
            else:
                return 1

        def get_score(gs, pred):
            return (1 - p) * np.sum([relevances[y_i] * estimated_probs[y_pred_i] for y_i, y_pred_i in zip(gs, pred)])

        # Commonly adopted probability value
        p = 0.8

        # Obtain the estimated position per class given the predictions (Toxicity ranking with higher toxicity on top)
        labels = [0, 1, 2, 3]
        pred_cs = [list(y_pred).count(label) for label in labels]

        estimated_probs = [get_estimated_prob(label, pred_cs) for label in labels]
        relevances = [get_relevance(label) for label in labels]

        return get_score(y, y_pred)

    @staticmethod
    def _pearson_coef(y, y_pred):
        return pearsonr(y, y_pred)[0]

    @staticmethod
    def _check_inconsistencies(y, y_pred):
        first_inconsistency_msg = f'Number of predictions ({len(list(y_pred))}) do not match ' \
                                  f'the number of gold standard labels ({len(list(y))})'
        assert len(list(y)) == len(list(y_pred)), first_inconsistency_msg

        second_incosistency_msg = f'Prediction labels contain an invalid label ' \
                                  f'(valid_labels: [0, 1, 2, 3], predicted_labels: {list(set(y_pred))})'
        assert np.alltrue(np.isin(y_pred, [0, 1, 2, 3])), second_incosistency_msg

    def evaluate(self, y, y_pred):
        self._check_inconsistencies(y, y_pred)
        return {'f1_toxic': self._f1_toxic(np.array(y) > 0, np.array(y_pred) > 0),
                'cem': self._cem_score(y, y_pred),
                'rbp': self._rbp(y, y_pred),
                'pearson': self._pearson_coef(y, y_pred)}
