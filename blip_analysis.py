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
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.gridspec import GridSpec
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import exp_blip_models_manu as em
from sklearn.model_selection import train_test_split
sys.path.append('/home/camp/warner/working/Recordings/binary_pulses')



def load_recs(sb=False):
    ### Accuracy wont work because more similar responses have a higher mistake therefore i need to flip the scale. 
    rec1 = br.Binary_recording('../200228/2020-02-28_19-56-29/', 32, '../200228/2020-02-28trial_names_ventral.txt', sniff_basis=sb)
    rec2 = br.Binary_recording('../200228/2020-02-28_16-37-36/', 32, '../200228/2020-02-28trial_names_dorsal.txt', sniff_basis=sb)
    rec3 = br.Binary_recording('../200303/2020-03-03_16-44-23/', 32, '../200303/2020-03-03trial_names_dorsal.txt', sniff_basis=sb)
    rec4 = br.Binary_recording('../200303/2020-03-03_19-57-03/', 32, '../200303/2020-03-03trial_names_ventral.txt', sniff_basis=sb)
    rec5 = br.Binary_recording('../200309/2020-03-09_16-20-42/', 32, '../200309/2020-03-09trial_name_joined.txt', sniff_basis=sb)
    rec6 = br.Binary_recording('../200311/2020-03-11_16-51-10/', 32, '../200311/2020-03-11trial_name_binary_joined.txt', sniff_basis=sb)
    rec7 = br.Binary_recording('../200318/2020-03-18_15-24-43/', 32, '../200318/2020-03-18trial_name.txt', sniff_basis=sb)
    rec8 = br.Binary_recording('../200319/2020-03-19_16-08-45/', 32, '../200319/2020-03-19_16-08-45_trial_names.txt', sniff_basis=sb)

    recs = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])
    return recs

if 'recs' not in globals():
    recs= load_recs()
else:
    print('Found recs already')

if 'mean_resps' not in globals():
    mean_resps = {1:None, 3:None, 5:None}

if 'mean_resps_bl' not in globals():
    mean_resps_bl = {1:None, 3:None, 5:None}
if 'mean_resps_sb' not in globals():
    mean_resps_sb = {1:None, 3:None, 5:None}

if 'units_usrts' not in globals():
    units_usrts = {1:None, 3:None, 5:None}

if 'odour_variances' not in globals():
    odour_variances = {1:None, 3:None, 5:None}

def get_trial_array():
    return np.array([[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)])


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

def response_plot(mean_responses, cell_index: int, return_shifts: bool = False, pred_funcs=None, return_argmaxsum: bool =False, sharey: bool =False, return_axis=False, plot_argmaxsum=True, pred_func_labels=None):
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
    

def dendo_and_heatmap(bin_weights, fig=None, return_fig = False):
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    bin_weights = np.array(bin_weights)
    Z = linkage(bin_weights, method='complete', optimal_ordering=True)
    gs = GridSpec(2, 20)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[1, :-1])
    cbar = fig.add_subplot(gs[1, -1])
    d = dendrogram(Z, ax=ax1);
    ax1.set_xticks([])
    im = ax2.imshow(bin_weights[d['leaves']].T, aspect='auto', cmap='seismic', vmin=-np.max(abs(bin_weights)), vmax=np.max(abs(bin_weights)))
    plt.colorbar(mappable=im, cax=cbar)
    if return_fig:
        return fig
    
def get_usrts(odour_index, **kwargs):
    if 'pre_trial_window' not in kwargs:
        kwargs['pre_trial_window'] = 0
    if 'post_trial_window' not in kwargs:
        kwargs['post_trial_window'] = 0.38
    if units_usrts[odour_index] is None:
        resps = [recs.get_binned_trial_response(f'{i}_{odour_index}', **kwargs)[1] for i in range(32)]
        units_usrt = np.array([[repeation for recording in trial for repeation in np.rollaxis(recording, 1)] for trial in resps]).T
        units_usrts[odour_index] = units_usrt
    else:
        units_usrt = units_usrts[odour_index]
    return units_usrt

def get_variances(odour_index):
    if units_usrts[odour_index] is None:
        print('Calculating usrt')
    units_usrt = get_usrts(odour_index)
    all_vars = []
    for i in range(len(units_usrt)):
        mean_resps = [[np.sum(k) for k in j] for j in units_usrt[i]]
        all_vars.append([np.var(j) for j in mean_resps])

    odour_variances[odour_index] = all_vars
    return all_vars
    
    
def get_stable_resp_indexes(odour_index='all'):
    if odour_index=='all':
        all_stables = []
        for i in [1, 3, 5]:
            all_vars = get_variances(i)
            stable_indexes = np.arange(len(all_vars))[np.min(np.array(all_vars), axis=-1) != 0]
            all_stables.append(stable_indexes)
        stable_indexes = np.array([i for i in range(145) if i in all_stables[0] and i in all_stables[1] and i in all_stables[2]])
    else:
        all_vars = get_variances(odour_index)
        stable_indexes = np.arange(len(all_vars))[np.min(np.array(all_vars), axis=-1) != 0]
    return stable_indexes

def split_data_train(units_usrt, test_size=0.5, model_type=None):
    if model_type is None:
        model_type = em.ExponentialInteractiveModel
    
    units_usrt_split = np.array([[train_test_split(i, test_size=test_size) for i in j] for j in units_usrt])
    units_usrt_train = units_usrt_split[:, :, 0]
    units_usrt_test = units_usrt_split[:, :, 1]
    models_train = []
    models_test = []
    
    for i in range(len(units_usrt)):
        model = model_type(units_usrt_train, i)
        model.fit()
        models_train.append(model)

        model = model_type(units_usrt_test, i)
        model.fit()
        models_test.append(model)
        
    return models_train, models_test

def subtract_pc(models_train, models_test, pc_index=0, subtract_mean=True, normalise=True, inv_bins_index=None):
    bins_train = np.array([i.opt_out.x[:-1] for i in models_train])
    bins_test = np.array([i.opt_out.x[:-1] for i in models_test])
    if inv_bins_index is None:
        inv_bins_index = [np.sign(i)[0] for i in bins_train]
    inv_bins_index = np.array(inv_bins_index)
    inv_bins_train = bins_train * inv_bins_index[:, np.newaxis]
    inv_bins_test = bins_test * inv_bins_index[:, np.newaxis]
    pca_split = PCA(n_components=9)
    pcad_inv_bins_train = pca_split.fit_transform(inv_bins_train)
    pcad_inv_bins_test = pca_split.transform(inv_bins_test)
    pcad_inv_bins_train[:,  0] = np.zeros(len(pcad_inv_bins_train))
    pcad_inv_bins_test[:,  0] = np.zeros(len(pcad_inv_bins_train))
    if subtract_mean:
        pcad_inv_bins_train = pca_split.inverse_transform(pcad_inv_bins_train) - pca_split.mean_
        pcad_inv_bins_test = pca_split.inverse_transform(pcad_inv_bins_test) - pca_split.mean_
    if normalise:
        pcad_inv_bins_train = pcad_inv_bins_train/ np.max(np.abs(pcad_inv_bins_train), axis=-1)[:, np.newaxis]
        pcad_inv_bins_test = pcad_inv_bins_test/ np.max(np.abs(pcad_inv_bins_test), axis=-1)[:, np.newaxis]
    
    return pcad_inv_bins_train, pcad_inv_bins_test

def cluster_and_score_train_test(train_bins, test_bins, cluster_type = None, **kwargs):
    if cluster_type is None:
        cluster_type = AgglomerativeClustering
        if 'affinity' not in kwargs:
            kwargs['affinity'] = 'euclidean'
        if 'linkage' not in kwargs:
            kwargs['linkage'] = 'complete'
    clustering = cluster_type(**kwargs)
    cluster_train = clustering.fit_predict(train_bins)
    cluster_test = clustering.fit_predict(test_bins)
#    print(cluster_train)
    if any(cluster_train == -1):
        #print('a')
        non_noise_cluster_train = cluster_train[np.where(cluster_train != -1)[0]]
        non_noise_cluster_test = cluster_test[np.where(cluster_test != -1)[0]]
        non_noise_train_bins = train_bins[np.where(cluster_train != -1)[0]]
        non_noise_test_bins = test_bins[np.where(cluster_test != -1)[0]]
        #print(non_noise_cluster_train, non_noise_cluster_test)
        if len(np.unique(non_noise_cluster_train)) == 1 or len(np.unique(non_noise_cluster_test)) == 1:
            train_silh = 0
            test_silh = 0
        else:
            train_silh = silhouette_score(non_noise_train_bins, non_noise_cluster_train)
            test_silh = silhouette_score(non_noise_test_bins, non_noise_cluster_test)
    else:
        train_silh = silhouette_score(train_bins, cluster_train)
        test_silh = silhouette_score(test_bins, cluster_test)
    randi = adjusted_rand_score(cluster_train, cluster_test)
    return train_silh, test_silh, randi