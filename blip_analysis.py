
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
import pdb
from copy import deepcopy
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('/home/camp/warner/working/Recordings/binary_pulses')



def load_recs(sb=False) -> jr.JoinedRecording:
    ### Accuracy wont work because more similar responses have a higher mistake therefore i need to flip the scale. 
    rec1: br.Binary_recording = br.Binary_recording('200228/2020-02-28_19-56-29', 32, '200228/2020-02-28trial_names_ventral.txt', sniff_basis=sb)
    rec2: br.Binary_recording = br.Binary_recording('200228/2020-02-28_16-37-36/', 32, '200228/2020-02-28trial_names_dorsal.txt', sniff_basis=sb)
    rec3: br.Binary_recording = br.Binary_recording('200303/2020-03-03_16-44-23/', 32, '200303/2020-03-03trial_names_dorsal.txt', sniff_basis=sb)
    rec4: br.Binary_recording = br.Binary_recording('200303/2020-03-03_19-57-03/', 32, '200303/2020-03-03trial_names_ventral.txt', sniff_basis=sb)
    rec5: br.Binary_recording = br.Binary_recording('200309/2020-03-09_16-20-42/', 32, '200309/2020-03-09trial_name_joined.txt', sniff_basis=sb)
    rec6: br.Binary_recording = br.Binary_recording('200311/2020-03-11_16-51-10/', 32, '200311/2020-03-11trial_name_binary_joined.txt', sniff_basis=sb)
    rec7: br.Binary_recording = br.Binary_recording('200318/2020-03-18_15-24-43/', 32, '200318/2020-03-18trial_name.txt', sniff_basis=sb)
    rec8: br.Binary_recording = br.Binary_recording('200319/2020-03-19_16-08-45/', 32, '200319/2020-03-19_16-08-45_trial_names.txt', sniff_basis=sb)

    recs: jr.JoinedRecording = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])
    return recs


# if 'recs' not in globals():
#     recs= load_recs()
# else:
#     print('Found recs already')

if 'mean_resps' not in globals():
    mean_resps = {1:None, 3:None, 5:None}

if 'mean_resps_bl' not in globals():
    mean_resps_bl = {1:None, 3:None, 5:None}
if 'mean_resps_sb' not in globals():
    mean_resps_sb = {1:None, 3:None, 5:None}

def get_recs() -> jr.JoinedRecording:
    return recs

def load_sb_recs():
    if 'sb_recs' not in globals():
        global sb_recs 
        sb_recs = load_recs(sb=True)
        return sb_recs

def _generate_mean_resps(odour_index, bl=False, sb=False):
    if not sb:
        data= [recs.get_binned_trial_response('%d_%d' % (m, odour_index), pre_trial_window=1, post_trial_window=1, baselined=bl)[1] for m in range(32)]
    else:
        load_sb_recs()
        data = [sb_recs.get_binned_trial_response('%d_%d' % (m, odour_index), pre_trial_window=3, post_trial_window=3, baselined=False, bin_size=0.05)[1] for m in range(32)]
    data2 = [[np.array(data[i][j]).swapaxes(0, 1) for j in range(len(data[i]))] for i in range(32)]
    data3= [[i for j in k for i in j] for k in data2]
    if not sb:
        if bl:
            mean_resps_bl[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])
        else:
            mean_resps[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])
    else:
        mean_resps_sb[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])
    

def _generate_unit_func(unit_index, func, bl=False, data=None, **kwargs):
    if data is None:
        data= [recs.get_binned_trial_response('%d_%d' % (m, odour_index), pre_trial_window=1, post_trial_window=-0.12, baselined=bl)[1] for m in range(32)]
    data2 = [[np.array(data[i][j]).swapaxes(0, 1) for j in range(len(data[i]))] for i in range(32)]
    data3= [[i for j in k for i in j] for k in data2]
    var_resps = np.array([[func(cell, **kwargs) for cell in trial]for trial in data3])


def return_mean_resp(odour, bl=False, sb=False):
    if not bl:
        if not sb:
            if odour in mean_resps:
                if mean_resps[odour] is None:
                    print('Generating mean odour %d response' % odour)
                    _generate_mean_resps(odour, bl=bl)
                return mean_resps[odour]
            else:
                raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')
        else:
            if odour in mean_resps_sb:
                if mean_resps_sb[odour] is None:
                    print('Generating mean odour %d response' % odour)
                    _generate_mean_resps(odour, bl=bl, sb=sb)
                return mean_resps_sb[odour]
            else:
                raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')
    else:
        if sb:
            print('No sniff basis basline subtracted sorry.')
        if odour in mean_resps_bl:
            if mean_resps_bl[odour] is None:
                print('Generating baselined mean odour %d response' % odour)
                _generate_mean_resps(odour, bl=bl)
            return mean_resps_bl[odour]
        else:
            raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')

def response_plot(mean_responses, cell_index: int, return_shifts: bool = False, pred_funcs=None, return_argmaxsum: bool =False, sharey: bool =False, return_axis=False, plot_argmaxsum=True, pred_func_labels=None) -> Optional[list]:
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
    ax[1].plot(arg_maxes_sum, 'o-', alpha=plot_argmaxsum)
    ax[1].set_xticks(range(32))
    ax[1].set_xticklabels(labels=[f'{i:05b}' for i in range(32)], rotation=90);
    ax[1].grid(True)
    if not sharey:
        ax2 = ax[1].twinx()
    else:
        ax2 = ax[1]
    if pred_funcs is None:
        ax2.plot(np.max(unit[:, 100:], axis=-1), 'o-', color='r', label='peak_val')
    else:
        for index, pred_func in enumerate(pred_funcs):
            if pred_func_labels is None:
                label='pred_func %d' % index
            else:
                label=pred_func_labels[index]
            ax2.plot(pred_func, 'o-', label=label, color='C%d' % (index+1))
    ax2.legend()
    returned = []
    if return_argmaxsum:
        returned.append(arg_maxes_sum)
    if return_shifts:
        returned.append(avg_shifts)
    if return_axis:
        returned.append(ax)
    if len(returned) > 0:
        return returned
    
def find_peak_ints(mean_response, cell_index, stim_start=100, pre_peak_window=5, post_peak_window=15):
    unit = mean_response[:, cell_index]
    arg_maxes_mean = []
    for i in range(32):
        arg_max  = np.argmax(unit[i][stim_start:])
        arg_maxes_mean.append(np.mean(unit[i][stim_start+arg_max-pre_peak_window:stim_start+arg_max+post_peak_window]))
    return arg_maxes_mean


def sigmoidal_model(trial_int, *params):
    s = f'{trial_int:05b}'
    resp_val = sum([val * (i == '1') for val, i in zip(params[:5], s)])
    amp, thresh, bl = params[5:]
    resp_val = amp/(1+np.exp(-(resp_val + thresh))) + bl
    return resp_val

def sigmoidal_model2(trial_int, *params):
    s = f'{trial_int:05b}'
    resp_val = sum([val * (i == '1') for val, i in zip(params[:5], s)])
    resp_val += sum([val * (i+j == '11') for val, i, j in zip(params[5:], s[:-1],s[1:])])
    amp, thresh, bl = params[-3:]
    resp_val = amp/(1+np.exp(-(resp_val + thresh))) + bl
    return resp_val

def model_error(params, true_value, step_func):
    resp_vec = np.array([step_func(i, *params) for i in range(32)])
    return np.mean((resp_vec - true_value)**2)/np.var(true_value)  #+ abs(sum(params[:5]) - params[5])

def linear_reg_fit(arg_maxes, bl=None, amp=None, thresh=None, model=None, bl_mult=0.9, amp_mult=1.1, init_guess = None, bounds=None, **kwargs):
    if bl is None: bl = np.min(arg_maxes)*bl_mult
    if amp is None: amp = (np.max(arg_maxes) - bl)*amp_mult
    if thresh is None: thresh = 0
    if model is None: model = "sigmoidal_model"
    #pdb.set_trace()
    model_func = models[model]['func']
    bin_array = models[model]['bin_array']
    if init_guess is None:
        inter_log = (arg_maxes - bl)/(amp-arg_maxes + bl)
        inter_log = np.maximum(inter_log, 1e-5)
        wx = np.log(inter_log)

        lr = LinearRegression()
        lr.fit(bin_array, wx)
    
    
        # Final lr coef is the threshold
        print('init_guess', lr.coef_, lr.intercept_)
        params = list(lr.coef_) + [amp, lr.intercept_, bl]
    else:
        params = init_guess
    if 'method' not in kwargs:
        kwargs['method'] = 'L-BFGS-B'
    if bounds is None:
        bounds = [(None, None) for i in range(len(params))]
        bounds[:-3] = [(-5, 5)]*len(bounds[:-3])
        bounds[-3] = (0, amp*2)
        bounds[-1] = (0, None)
    min_fun = minimize(model_error, params, args=(arg_maxes, model_func), **kwargs, bounds=bounds)
    print(min_fun.fun, "\n")
    print(min_fun.message)
    print(min_fun.nit)
    return min_fun, min_fun.x


def z_score_plot(data, dataz,unit_id, pos_thresh, neg_thresh, figsize=(13, 6)):
    bin_reps = [[int(char) for char in f'{i:05b}'] for i in range(32)]
    gs = GridSpec(1, 10)
    fig = plt.figure(figsize=figsize)
    ax4 =plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1:4])
    ax2 = plt.subplot(gs[0, 4:7])
    ax3 = plt.subplot(gs[0, 7:])
    ax1.matshow(data[:, unit_id], aspect='auto', cmap='bwr', extent= [-1, 1.12, 31, 0], vmin=-np.max(abs(data[:, unit_id])), vmax=np.max(abs(data[:, unit_id])))


    ax2.matshow(relu2(dataz[:, unit_id], pos_thresh, neg_thresh), aspect='auto', cmap='bwr', extent= [-1, 1.12, 31, 0], vmin=-np.max(abs(dataz[:, unit_id])),  vmax=np.max(abs(dataz[:, unit_id])))
   
    ax3.plot(np.sum(relu_neg(dataz[:, unit_id], neg_thresh), axis=-1), range(32), label='neg', color='b')
    ax3.plot(np.sum(relu(dataz[:, unit_id], pos_thresh), axis=-1), range(32), label='pos', color='r')
    ax3.plot(np.sum(relu2(dataz[:, unit_id], pos_thresh, neg_thresh), axis=-1), range(32), label='sum', color='violet')

    ax3.legend()

    ax1.set_xlabel('Time (s)')
    #ax1.set_ylabel('Trial number')
    ax3.set_xlabel('Summed Z score')
    #ax3.set_xlim(0, 3)
    ax3.grid(True)
    vmin = np.min(data[:, unit_id])
    vmax = np.max(data[:, unit_id])
    vminz = np.min(dataz[:, unit_id])
    vmaxz = np.max(dataz[:, unit_id])
    fig.suptitle(f'Vmin_raw={vmin:.2f}, Vmax_raw={vmax:.2f}, Vmin_z={vminz:.2f}, Vmax_z={vmaxz:.2f}')
    ax4.imshow(bin_reps)

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax3.set_ylim(ax4.get_ylim())

    ax4.set_xticks([])
    ax4.set_ylabel('Trial number')
    

def compute_for_dimension(data, dimension, func):
    dimensions = np.arange(len(data.shape))
    dimensions[0] = dimension
    dimensions[dimension] = 0
    data_transposed = data.transpose(dimensions)
    data_reshape = data_transposed.reshape(data_transposed.shape[0], -1)
    data_funced = func(data_reshape, axis=-1)
    return data_funced
    

def z_score_resps(resp, bl_window=slice(0, 100), tol=1e-6, method='basic'):
    if method == 'basic':
        mean_resp = compute_for_dimension(resp[:, :, bl_window], 1, np.mean)
        std_resp  = compute_for_dimension(resp[:, :, bl_window], 1, np.std)
        z_scored_resp = (resp-mean_resp[np.newaxis, :, np.newaxis])/(std_resp[np.newaxis, :, np.newaxis]+tol)
        return z_scored_resp
    if method == 'seperate':
        resp_copy_pos = deepcopy(resp)
        resp_copy_pos[resp_copy_pos <= 0] = np.nan
        resp_copy_neg = deepcopy(resp)
        resp_copy_neg[resp_copy_neg > 0] = np.nan
        
        pos_mean_resp = compute_for_dimension(resp_copy_pos[:, :, bl_window], 1, np.nanmean)
        pos_std_resp  = compute_for_dimension(resp_copy_pos[:, :, bl_window], 1, np.nanstd)
        neg_mean_resp = compute_for_dimension(resp_copy_neg[:, :, bl_window], 1, np.nanmean)
        neg_std_resp  = compute_for_dimension(resp_copy_neg[:, :, bl_window], 1, np.nanstd)
        
        z_scored_pos = (resp_copy_pos-pos_mean_resp[np.newaxis, :, np.newaxis])/(pos_std_resp[np.newaxis, :, np.newaxis]+tol)
        z_scored_neg = (resp_copy_neg-neg_mean_resp[np.newaxis, :, np.newaxis])/(neg_std_resp[np.newaxis, :, np.newaxis]+tol)
        
        z_scored_resp = np.nan_to_num(z_scored_pos) + np.nan_to_num(z_scored_neg)
        return z_scored_resp, np.nan_to_num(z_scored_pos), np.nan_to_num(z_scored_neg)

relu= lambda x, th: (x-th)*(x>th)

relu2= lambda x, th_max, th_min: (x-th_max)*(x>th_max) -(x+th_min)*(x < -th_min)

relu_neg = lambda x, th: -(x+th)*(x<-th)

models = {
        'sigmoidal_model1':{'func':sigmoidal_model, 'bin_array':[[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)]},
        'sigmoidal_model2':{'func':sigmoidal_model2, 'bin_array':[[int(j) for j in f'{trial_int:05b}']+[(i+j == '11') for i, j in zip(f'{trial_int:05b}'[:-1], f'{trial_int:05b}'[1:])] for trial_int in range(32)]}
         }
