import numpy as np
import scipy
from sklearn.model_selection import StratifiedShuffleSplit

class ExponentialModel():
    '''
    LNP models using exponential non-linearity. 
    '''
    trial_array = np.array([[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)])

    def __init__(self, units_usrt, unit_id, stim_count_type='mean'):
        '''
        Inputs:
        units_usrt - The firing rate responses, in a tensor of unit x stimuli type x repeat x time
        unit_id - The index of the unit this model will be fitting to

        unit_srt - The full response for the cell to be fit to
        unit_sr - The average firing rate of this cell across all stimuli and repeats
        unit_sr_flat - Each average firing rate across all stimuli and repeats, flatterned into one array
        unit_sr_var - The variance of the cell's response across the repeats of each stimulus
        trial_array_full - Generates an extended list of trial input patterns, according to the number of repeats of each pattern
        is_fit - Has this been fit to the data yet
        fit_score - Misnomar - the fitting error
        true_resp - The true average firing rate change
        '''
        self.unit_srt = units_usrt[unit_id]
        if stim_count_type == 'mean':
            self.unit_sr = [[np.mean(r) for r in s] for s in self.unit_srt] # Should be mean as it units_srt is in firing rate not spike count
        elif stim_count_type == 'sum':
            self.unit_sr = [[np.sum(r) for r in s] for s in self.unit_srt]
        self.unit_sr_flat = np.maximum([r for s in self.unit_sr for r in s], 1e-5)
        self.unit_sr_var = np.maximum(np.array([np.power(np.std(i), 2) for i in self.unit_sr]), 1e-5)
        self.unit_id = unit_id
        self.trial_array_full = np.concatenate([[list(self.trial_array[s])]*len(self.unit_sr[s]) for s in range(len(self.trial_array))])
        self.is_fit = False
        self.fit_score = np.inf
        self.loss_val = np.inf
    
        self.true_resp = [np.mean(s) for s in self.unit_sr]
    
    
    def minimisation_loss(self, w, X=None, y_1=None):
        '''
        Calculate the loss for use in the minimisation function.
        '''

        # If no X and y_1 are presented (as is the case when its being implemented in the minimize function) then take the resps
        if X is None: X = self.true_resp
        if y_1 is None: y_1 = np.append(self.trial_array, np.ones((len(self.trial_array), 1)), axis=1)
        assert len(w) == y_1.shape[1]
        pred_response = np.exp(w@y_1.T)
        if any(np.isinf(pred_response)):
            print('a')
        return self.loss(pred_response, X)
    
    def loss(self, pred_response, true_response):
        '''
        Calculates the loss of a Poisson firing process model
        '''
        return np.sum(pred_response-true_response*np.log(pred_response))
    
    def fit_scores(self, true_resp=None, pred_resp=None, vars=None):
        '''
        Returns the mean square difference between the predicted and actual firing rates
        '''
        if true_resp is None: true_resp = self.true_resp
        if pred_resp is None: pred_resp= self.pred_resp
        if vars is None: vars = self.unit_sr_var
        return np.array([np.power((i-j), 2) for i, j in zip(true_resp, pred_resp)])/vars

    def fit(self, X=None, y=None, W=None, update_loss=True):
        '''
        Main fitting function, uses scipy minimize function on the loss function
        '''

        # As with the minimisation_loss and fit_score functions, if no parameters are passed, take the default ones
        opt_out = None
        if X is None: X = self.true_resp
        if y is None: y = self.trial_array

        # The weights can be passed to the function, in which case it will not attempt to find the best fit weighting.
        if W is None:
            y_1 = np.append(y, np.ones((len(y), 1)), axis=1)
            opt_out = scipy.optimize.minimize(self.minimisation_loss, np.zeros(len(y_1[0])), args=(X, y_1))
            W = opt_out.x
            self.opt_out= opt_out
        inter_exp = y@W[:-1]+W[-1] # The value inside the exponetial
        pred_firing = np.exp(inter_exp)
        self.is_fit = True
        self.pred_resp = pred_firing
        self.fit_score = np.mean(self.fit_scores())
        if update_loss:
            self.loss_val = self.loss(self.pred_resp, self.true_resp)
        
    
    def fit_split(self, n_splits=100, test_size=0.5, train_test_var=False, random_state=None, sss=None):
        '''
        Fit portions of the data and compare fitted values across different splits
        '''
        if sss is None:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        trial_labels =  np.concatenate([[s for j in range(len(self.unit_sr[s]))] for s in range(len(self.trial_array))])
        all_train_scores = []
        all_test_scores = []
        all_X_test_avg = []
        all_X_train_avg = []
        all_pred_test_avg = []
        all_pred_train_avg = []
        for train_index, test_index in sss.split(self.unit_sr_flat, trial_labels):
            X_train = self.unit_sr_flat[train_index]
            X_test = self.unit_sr_flat[test_index]
            y_train = self.trial_array_full[train_index]
            y_test = self.trial_array_full[test_index]
            labels_train = trial_labels[train_index]
            labels_test = trial_labels[test_index]
            self.fit(X_train, y_train, update_loss=False)
            out_train, pred_train = self.opt_out, self.pred_resp
            self.fit(X_test, y_test, W=out_train.x, update_loss=False)
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
            all_X_train_avg.append(np.array(X_train_avg))
            all_X_test_avg.append(np.array(X_test_avg))
            all_pred_train_avg.append(np.array(pred_train_avg))
            all_pred_test_avg.append(np.array(pred_test_avg))
            if train_test_var:
                all_train_scores.append(np.array(train_scores)/np.array(train_unit_var))
                all_test_scores.append(np.array(test_scores)/np.array(test_unit_var))
            
            else:
                all_train_scores.append(np.array(train_scores)/self.unit_sr_var)
                all_test_scores.append(np.array(test_scores)/self.unit_sr_var)
        self.train_scores = np.array(all_train_scores)
        self.test_scores = np.array(all_test_scores)
        self.X_train_avg = np.array(all_X_train_avg)
        self.X_test_avg = np.array(all_X_test_avg)
        self.pred_train_avg = np.array(all_pred_train_avg)
        self.pred_test_avg = np.array(all_pred_test_avg)
        

                
            


class ExponentialInteractiveModel(ExponentialModel):
    trial_array = np.array([[int(j) for j in f'{trial_int:05b}']+[(i+j == '11') for i, j in zip(f'{trial_int:05b}'[:-1], f'{trial_int:05b}'[1:])] for trial_int in range(32)])

    def __init__(self, unit_usrt, unit_id, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type )
            
class ExponentialCustomTrialArray(ExponentialModel):

    def __init__(self, unit_usrt, unit_id, trial_array, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type)
        self.trial_array = trial_array
        self.trial_array_full = np.concatenate([[list(self.trial_array[s])]*len(self.unit_sr[s]) for s in range(len(self.trial_array))])

class ExponentialJoinedWeights(ExponentialInteractiveModel):
    
    def __init__(self, unit_usrt, unit_id, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type)
        
    
class ExponentialSetWeighting(ExponentialInteractiveModel):
    
    def __init__(self, unit_usrt, unit_id, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type)
        
    def fit(self, W, X=None, y=None):
        at = np.ones(2)
        opt_out = scipy.optimize.minimize(self.minimisation_loss, at, args=(W))
        self.opt_out = opt_out
        at = opt_out.x
        w = W*at[0]
        wt = np.append(w, at[1])
        super().fit(W=wt)
        self.loss_val = opt_out.fun

    
    def minimisation_loss(self, at, W=np.ones(9)):
        w = W*at[0]
        wt = np.append(w, at[1])
        #print(wt)
        return super().minimisation_loss(wt)
    
class ExponentialPartSetWeighting(ExponentialInteractiveModel):
    
    def __init__(self, unit_usrt, unit_id, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type)
        
    def fit(self, W, X=None, y=None):
        ati = np.ones(6)
        opt_out = scipy.optimize.minimize(self.minimisation_loss, ati, args=(W))
        self.opt_out = opt_out
        ati = opt_out.x
        w_time = W*ati[0]
        w = np.append(w_time, ati[-4:])
        wt = np.append(w, ati[1])
        super().fit(W=wt)
        self.loss_val = opt_out.fun

    
class JoinedWeightModels():
     
    def __init__(self, unit_usrt, unit_ids, stim_count_type='mean'):
        self.models = [ExponentialJoinedWeights(unit_usrt, i, stim_count_type) for i in unit_ids]
        
    def fit_all(self, X=None, y=None, W=None, method='Nelder-Mead', alpha=1, **kwargs):
        if W is None:
            W = np.ones(len(self.models[0].trial_array[0]))
        at = np.ones(len(self.models)*2)
        wat = np.append(W, at)
        opt_out = scipy.optimize.minimize(self.minimisation_loss, wat, method = method, options={'maxiter':100000000}, args=(alpha))
        self.opt_out = opt_out
        self.loss_val = opt_out.fun
        

        
    def minimisation_loss(self, wat, alpha=1):
        losses = []
        w = wat[:9]
        ats = wat[9:] 
        for index, i in enumerate(self.models):
            w_full = np.array(list(ats[index]*w) + list([ats[index+len(self.models)]]))
            #print(w_full)
            loss = i.minimisation_loss(w_full)
            #print(loss)
            #print(w_full.shape)
            losses.append(loss)
        #print(np.sum(losses))
        #print(np.mean(losses) + np.mean(abs(ats)))
        return np.mean(losses) + np.mean(abs(ats))*alpha
    

class ExponentialRespWeighting(ExponentialInteractiveModel):
    def __init__(self, unit_usrt, unit_id, stim_count_type='mean'):
        super().__init__(unit_usrt, unit_id, stim_count_type)
        self.wt = np.ones(9)
        
    def fit(self, W_time, W_int, X=None, y=None):
        amps_and_thresh = np.ones(3)
        opt_out = scipy.optimize.minimize(self.minimisation_loss, amps_and_thresh, args=(W_time, W_int))
        self.opt_out = opt_out
        amps_and_thresh = opt_out.x
        w_time = W_time * amps_and_thresh[0]
        w_int = W_int * amps_and_thresh[1]
        w = np.append(w_time, w_int)
        wt = np.append(w, amps_and_thresh[-1])
       
        self.wt = wt
        super().fit(W=wt)
        self.loss_val = opt_out.fun
        
    def minimisation_loss(self, amps_and_thresh, W_time = np.ones(5), W_int = np.ones(4)):
        w_time = W_time*amps_and_thresh[0]
        w_int = W_int*amps_and_thresh[1]
        w = np.append(w_time, w_int)
        wt = np.append(w, amps_and_thresh[-1])
        return super().minimisation_loss(wt)
    
