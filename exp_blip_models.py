import numpy as np
import scipy
from sklearn.model_selection import StratifiedShuffleSplit


class ExponentialModel():
    trial_array = np.array([[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)])

    def __init__(self, units_usrt, unit_id):
        self.unit_srt = units_usrt[unit_id]
        self.unit_sr = [[np.mean(r) for r in s] for s in self.unit_srt]
        self.unit_sr_flat = np.maximum([r for s in self.unit_sr for r in s], 1e-5)
        self.unit_sr_var = np.maximum(np.array([np.power(np.std(i), 2) for i in self.unit_sr]), 1e-5)
        self.unit_id = unit_id
        self.trial_array_full = np.concatenate([[list(self.trial_array[s])]*len(self.unit_sr[s]) for s in range(len(self.trial_array))])
        self.is_fit = False
        self.fit_score = np.inf
        self.true_resp = [np.mean(s) for s in self.unit_sr]
    
    def minimisation_loss(self, w, X=None, y_1=None):
        if X is None: X = self.true_resp
        if y_1 is None: y_1 = np.append(self.trial_array, np.ones((len(self.trial_array), 1)), axis=1)
        assert len(w) == y_1.shape[1]
        pred_response = np.exp(w@y_1.T)
        return self.loss(pred_response, X)
    
    def loss(self, pred_response, true_response):
        return np.sum(pred_response-true_response*np.log(pred_response))
    

    def fit(self, X=None, y=None, W=None):
        opt_out = None
        if X is None: X = self.true_resp
        if y is None: y = self.trial_array
        if W is None:
            y_1 = np.append(y, np.ones((len(y), 1)), axis=1)
            opt_out = scipy.optimize.minimize(self.minimisation_loss, np.zeros(len(y_1[0])), args=(X, y_1))
            W = opt_out.x
        inter_exp = y@W[:-1]+W[-1]
        pred_firing = np.exp(inter_exp)
        self.is_fit = True
        self.pred_resp = pred_firing
        self.opt_out= opt_out
        
    
    def fit_split(self, n_splits=100, test_size=0.5, train_test_var=False, random_state=None, sss=None):
        if sss is None:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        trial_labels =  np.concatenate([[s for j in range(len(self.unit_sr[s]))] for s in range(len(self.trial_array))])
        all_train_scores = []
        all_test_scores = []
        for train_index, test_index in sss.split(self.unit_sr_flat, trial_labels):
            X_train = self.unit_sr_flat[train_index]
            X_test = self.unit_sr_flat[test_index]
            y_train = self.trial_array_full[train_index]
            y_test = self.trial_array_full[test_index]
            labels_train = trial_labels[train_index]
            labels_test = trial_labels[test_index]
            self.fit(X_train, y_train)
            out_train, pred_train = self.opt_out, self.pred_resp
            self.fit(X_test, y_test, W=out_train.x)
            pred_test =self.pred_resp

            X_train_avg = []
            X_test_avg = []
            pred_train_avg = []
            pred_test_avg = []
            train_unit_var = []
            test_unit_var = []

            for label in range(len(self.trial_array)):
                X_train_avg.append(np.mean(X_train[labels_train == label]))
                X_test_avg.append(np.mean(X_test[labels_test == label]))

                pred_train_avg.append(np.mean(pred_train[labels_train == label]))
                pred_test_avg.append(np.mean(pred_test[labels_test == label]))

                train_unit_var.append(np.std(X_train[labels_train==label])**2)
                test_unit_var.append(np.std(X_test[labels_test==label])**2)
            
            train_scores = np.array([np.power((i-j), 2) for i, j in zip(X_train_avg, pred_train_avg)])
            test_scores = np.array([np.power((i-j), 2) for i, j in zip(X_test_avg, pred_test_avg)])
            if train_test_var:
                all_train_scores.append(1-np.array(train_scores)/np.array(train_unit_var))
                all_test_scores.append(1-np.array(test_scores)/np.array(test_unit_var))
            
            else:
                all_train_scores.append(1-np.array(train_scores)/self.unit_sr_var)
                all_test_scores.append(1-np.array(test_scores)/self.unit_sr_var)
        self.train_scores = np.array(all_train_scores)
        self.test_scores = np.array(all_test_scores)

                
            


class ExponentialInteractiveModel(ExponentialModel):
    trial_array = np.array([[int(j) for j in f'{trial_int:05b}']+[(i+j == '11') for i, j in zip(f'{trial_int:05b}'[:-1], f'{trial_int:05b}'[1:])] for trial_int in range(32)])

    def __init__(self, unit_usrt, unit_id, ):
        super().__init__(unit_usrt, unit_id)
