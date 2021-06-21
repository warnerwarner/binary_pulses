
from typing import Optional

from scipy.signal.ltisys import step
import binary_recording as br
import joined_recording as jr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.signal import correlate

def load_recs() -> jr.JoinedRecording:
    ### Accuracy wont work because more similar responses have a higher mistake therefore i need to flip the scale. 
    rec1: br.Binary_recording = br.Binary_recording('200228/2020-02-28_19-56-29', 32, '200228/2020-02-28trial_names_ventral.txt')
    rec2: br.Binary_recording = br.Binary_recording('200228/2020-02-28_16-37-36/', 32, '200228/2020-02-28trial_names_dorsal.txt')
    rec3: br.Binary_recording = br.Binary_recording('200303/2020-03-03_16-44-23/', 32, '200303/2020-03-03trial_names_dorsal.txt')
    rec4: br.Binary_recording = br.Binary_recording('200303/2020-03-03_19-57-03/', 32, '200303/2020-03-03trial_names_ventral.txt')
    rec5: br.Binary_recording = br.Binary_recording('200309/2020-03-09_16-20-42/', 32, '200309/2020-03-09trial_name_joined.txt')
    rec6: br.Binary_recording = br.Binary_recording('200311/2020-03-11_16-51-10/', 32, '200311/2020-03-11trial_name_binary_joined.txt')
    rec7: br.Binary_recording = br.Binary_recording('200318/2020-03-18_15-24-43/', 32, '200318/2020-03-18trial_name.txt')
    rec8: br.Binary_recording = br.Binary_recording('200319/2020-03-19_16-08-45/', 32, '200319/2020-03-19_16-08-45_trial_names.txt')

    recs: jr.JoinedRecording = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])
    return recs

if 'recs' not in globals():
    recs: jr.JoinedRecording = load_recs()
else:
    print('Found recs already')

if 'mean_resps' not in globals():
    mean_resps = {1:None, 3:None, 5:None}

def get_recs() -> jr.JoinedRecording:
    return recs

def _generate_mean_resps(odour_index):
    data= [recs.get_binned_trial_response('%d_%d' % (m, odour_index), pre_trial_window=1, post_trial_window=1)[1] for m in range(32)]
    data2 = [[np.array(data[i][j]).swapaxes(0, 1) for j in range(len(data[i]))] for i in range(32)]
    data3= [[i for j in k for i in j] for k in data2]
    mean_resps[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])

def return_mean_resp(odour: int):
    if odour in mean_resps:
        if mean_resps[odour] is None:
            print('Generating mean odour %d response' % odour)
            _generate_mean_resps(odour)
        return mean_resps[odour]
    else:
        raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')

def response_plot(mean_responses, cell_index: int, return_shifts: bool = False, pred_funcs=None, return_argmaxsum: bool =False, sharey: bool =False, ) -> Optional[list]:
    unit = mean_responses[:, cell_index]
    fig, ax = plt.subplots(1, 2, sharey=False, figsize=(12, 4))
    arg_maxes_sum: list = []
    shifts = np.array([[np.argmax(correlate(unit[i][100:], unit[j][100:])) for i in range(32)] for j in range(32)])
    avg_shifts = [np.mean(shifts[:, i]) - 111 for i in range(32)]
    for i in range(32):
        ax[0].plot(np.arange(-1000-avg_shifts[i]*10, 1120-10*avg_shifts[i], 10), unit[i], color=cm.plasma(i/32))
        arg_max: int = np.argmax(unit[i][100:])
        arg_maxes_sum.append(np.mean(unit[i][100+arg_max-5:100+arg_max+15]))
    avg_shifts = np.array(avg_shifts)*-10
    ax[1].plot(arg_maxes_sum, 'o-')
    ax[1].set_xticks(range(32))
    ax[1].set_xticklabels(labels=[f'{i:05b}' for i in range(32)], rotation=90);
    ax[1].grid(True)
    if not sharey:
        ax2 = ax[1].twinx()
    else:
        ax2 = ax[1]
    if pred_funcs is None:
        ax2.plot(np.max(unit[:, 100:], axis=-1), 'o-', color='r')
    else:
        for index, pred_func in enumerate(pred_funcs):
            ax2.plot(pred_func, 'o-', label='pred_func %d' % index, color='C%d' % (index+1))
        ax2.legend()
    returned = []
    if return_argmaxsum:
        returned.append(arg_maxes_sum)
    if return_shifts:
        returned.append(avg_shifts)
    if len(returned) > 0:
        return returned
    
def find_peak_ints(mean_response, cell_index, stim_start=100, pre_peak_window=5, post_peak_window=15):
    unit = mean_response[:, cell_index]
    arg_maxes_mean = []
    for i in range(32):
        arg_max  = np.argmax(unit[i][stim_start:])
        arg_maxes_mean.append(np.mean(unit[i][stim_start+arg_max-pre_peak_window:stim_start+arg_max+post_peak_window]))
    return arg_maxes_mean


def step_model5(trial_int, *params):
    s = f'{trial_int:05b}'
    resp_val = sum([val * (i == '1') for val, i in zip(params[:5], s)])
    amp, thresh, bl = params[5:]
    resp_val = amp/(1+np.exp(-(resp_val + thresh))) + bl
    return resp_val

def step_model_error(params, true_value, step_func):
    resp_vec = np.array([step_func(i, *params) for i in range(32)])
    return np.mean((resp_vec - true_value)**2)/np.var(true_value)  #+ abs(sum(params[:5]) - params[5])

def linear_reg_fit(arg_maxes, bl=None, amp=None, thresh=None, step_model=None, bl_mult=0.9, amp_mult=1.1, **kwargs):
    if bl is None: bl = np.min(arg_maxes)*bl_mult
    if amp is None: amp = (np.max(arg_maxes) - bl)*amp_mult
    if thresh is None: thresh = 0
    if step_model is None: step_model = step_model5

    wx = np.log((arg_maxes - bl)/(amp-arg_maxes + bl))

    bin_array = [[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)]  #Additional 1 for threshold

    #print(bin_array)
    lr = LinearRegression()
    lr.fit(bin_array, wx)
    # Final lr coef is the threshold
    print(lr.coef_, lr.intercept_)
    params = list(lr.coef_) + [amp, lr.intercept_, bl]
    if 'method' not in kwargs:
        kwargs['method'] = 'CG'
    bounds = [(None, None) for i in range(8)]
    bounds[-1] = (0, None)
    min_fun = minimize(step_model_error, params, args=(arg_maxes, step_model), **kwargs, bounds=bounds)
    print(min_fun.fun, "\n")
    print(min_fun.message)
    print(min_fun.nit)
    return min_fun, min_fun.x
        
