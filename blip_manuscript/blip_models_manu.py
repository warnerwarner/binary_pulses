from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize

def check_not_none(iterable):
    return not any([i is None for i in iterable])
    

class LinearSigmoidModel():
    all_bin_array = np.array([[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)])
    num_trials, num_weights = all_bin_array.shape
    _estimator_type_ = "regressor"
    
    @classmethod
    def _predict(cls, bin_array, params):
        bin_weights, amp, thresh, bl = cls._params_to_vars(params)
        resp_val = bin_array@bin_weights
        resp_val = amp/(1+np.exp(-(resp_val + thresh))) + bl
        return resp_val

    @classmethod
    def _error(cls, X, y, params):
        resp_vec = cls._predict(X, params)
        return np.sum(resp_vec-y*np.log(resp_vec))
#        return np.mean((resp_vec -y)**2)/np.var(y)
    
    @classmethod
    def _params_to_vars(cls, params):
        *bin_weights, amp, thresh, bl = params
        return bin_weights, amp, thresh, bl
    
    def __init__(self, bl_mult=0.9, amp_mult=1.1, weight_bounds=(-5, 5), amp_multi_bound=(0, 2), bl_bounds=(0, None), thresh_bounds=(None, None), verbose=False, method=None):
        if method == 'DE':
            if not all([check_not_none(weight_bounds), check_not_none(amp_multi_bound), check_not_none(bl_bounds), check_not_none(thresh_bounds)]):
                print(check_not_none(weight_bounds), check_not_none(amp_multi_bound), check_not_none(bl_bounds), check_not_none(thresh_bounds))
                raise ValueError('DE needs non None bounds')
        self.bin_weights = None
        self.bl_mult = bl_mult
        self.amp_mult = amp_mult
        self.model_score = None
        self.thresh = None
        self.amp = None
        self.bl = None
        self.is_fit = False
        self.weight_bounds = weight_bounds
        self.amp_multi_bound = amp_multi_bound
        self.bl_bounds = bl_bounds
        self.thresh_bounds = thresh_bounds
        self.bounds = None
        self.verbose=verbose
        self.method=method
        
    def __repr__(self):
        return f"""
Linear Model method={self.method},
             is_fit={self.is_fit},
           bl_multi={self.bl_mult},
           amp_mult={self.amp_mult},
             bounds={self.bounds},
        bin_weights={self.bin_weights},
             thresh={self.thresh},
                amp={self.amp},
                 bl={self.bl}
              score={self.model_score},
                    """


    def get_params(self, deep=True):
        params_dict = {
            'bl_mult':self.bl_mult,
            'amp_mult':self.amp_mult,
            'weight_bounds':self.weight_bounds,
            'amp_multi_bound':self.amp_multi_bound,
            'bl_bounds':self.bl_bounds,
            'thresh_bounds':self.thresh_bounds,
            'verbose':self.verbose,
            'method':self.method
        }
        return params_dict
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict(self, X):
        X = np.array(X)
        params = self.vars_to_params()
        return self._predict(X, params)
    
    
        
    def fit(self, X, y, init_params=None, **kwargs):
        bl = np.min(y)*self.bl_mult
        amp = (np.max(y) - bl)*self.amp_mult
        thresh = 0
        if init_params is None:
            inter_log = (y - bl)/(amp-y + bl)
            inter_log = np.maximum(inter_log, 1e-5)
            wx = np.log(inter_log)

            lr = LinearRegression()
            lr.fit(X, wx)
            
            # Final lr coef is the threshold
            if self.verbose:
                print('init_guess', lr.coef_, lr.intercept_)
            
            # Setting the object parameters
            self.bin_weights = lr.coef_
            self.amp = amp
            self.thresh = lr.intercept_
            self.bl = bl
            
            # Getting the parameters as a nice vector in order
            params = self.vars_to_params()
        else:
            params = init_params
        # Minimise needs parameters as the first argument so swap around with X, y
        swap_args = lambda p, X, y: self._error(X, y, p)
        self.bounds =[self.weight_bounds]*self.num_weights + [(self.amp_multi_bound[0]*amp, self.amp_multi_bound[1]*amp), self.thresh_bounds, self.bl_bounds]
        #print(X.shape, self.bounds)
        if self.method is None:

            min_fun = minimize(swap_args, params, args=(X, y), **kwargs, bounds=self.bounds)
        elif self.method == 'DE':

            min_fun = differential_evolution(swap_args, self.bounds, args=(X, y), **kwargs)
        else:
            raise ValueError("Unknown method")
        self.is_fit=True

        self.update_vars_from_params(min_fun.x)
        self.model_score = self.score(X, y)
        return min_fun
    
    def score(self, X, y):
        X = np.array(X)
        params = self.vars_to_params()
        return 1 - self._error(X, y, params)
    
    def vars_to_params(self):
        params = *self.bin_weights, self.amp, self.thresh, self.bl
        return params

    def update_vars_from_params(self, params):
        self.bin_weights, self.amp, self.thresh, self.bl = self._params_to_vars(params)
        

        
        
class InteractionSigmoidalModel(LinearSigmoidModel):
    all_bin_array = np.array([[int(j) for j in f'{trial_int:05b}']+[(i+j == '11') for i, j in zip(f'{trial_int:05b}'[:-1], f'{trial_int:05b}'[1:])] for trial_int in range(32)])
    num_trials, num_weights = all_bin_array.shape
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f"""
       Interaction model is_fit={self.is_fit},
                       bl_multi={self.bl_mult},
                       amp_mult={self.amp_mult},
                         bounds={self.bounds},
                    bin_weights={self.bin_weights},
                         thresh={self.thresh},
                            amp={self.amp},
                             bl={self.bl},
                          score={self.model_score}
                    """
        