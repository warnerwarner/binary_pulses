"""Contains collection of general analysis functions use in the analysis of blip recordings"""
import sys
import binary_recording as br
import joined_recording as jr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.signal import correlate
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import exp_blip_models as em
sys.path.append('/home/camp/warner/working/Recordings/binary_pulses')



def load_recs(sb = False):
    """Load in the recordings

    Args:
        sb (bool, optional): Should the recordings be loaded in the sniff basis. Defaults to False.

    Returns:
        JoinedRecording: All recordings joined together in a JoinedRecording objext
    """
    rec1 = br.Binary_recording('../200228/2020-02-28_19-56-29/',
                                32,
                               '../200228/2020-02-28trial_names_ventral.txt',
                               sniff_basis=sb)
    rec2 = br.Binary_recording('../200228/2020-02-28_16-37-36/',
                                32,
                                '../200228/2020-02-28trial_names_dorsal.txt',
                                sniff_basis=sb)
    rec3 = br.Binary_recording('../200303/2020-03-03_16-44-23/',
                                32,
                                '../200303/2020-03-03trial_names_dorsal.txt',
                                sniff_basis=sb)
    rec4 = br.Binary_recording('../200303/2020-03-03_19-57-03/',
                                32,
                                '../200303/2020-03-03trial_names_ventral.txt',
                                sniff_basis=sb)
    rec5 = br.Binary_recording('../200309/2020-03-09_16-20-42/',
                                32,
                                '../200309/2020-03-09trial_name_joined.txt',
                                sniff_basis=sb)
    rec6 = br.Binary_recording('../200311/2020-03-11_16-51-10/',
                                32,
                                '../200311/2020-03-11trial_name_binary_joined.txt',
                                sniff_basis=sb)
    rec7 = br.Binary_recording('../200318/2020-03-18_15-24-43/',
                                32,
                                '../200318/2020-03-18trial_name.txt',
                                sniff_basis=sb)
    rec8 = br.Binary_recording('../200319/2020-03-19_16-08-45/',
                                32,
                                '../200319/2020-03-19_16-08-45_trial_names.txt',
                                sniff_basis=sb)

    rec_array = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])
    return rec_array

# Load in the recordings straight away and set some dictionaries
sb_recs = None
if 'recs' not in globals():
    recs = load_recs()
else:
    print('Found recs already')


def load_sniff_recs():
    """Loads the recs in sniff basis
    """
    if sb_recs is None:
        sb_recs = load_recs(sb=True)


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
    """Get the default trial array (time and interaction bins)

    Returns:
        numpy.array: An array of the trial bin arrays
    """
    return np.array([[int(j) for j in f'{trial_int:05b}'] for trial_int in range(32)])


def _generate_mean_resps(odour_index: int, bl=False, sb=False):
    """Internal function. Generates the mean responses of cells.
       Needs to juggle the axes around due to the inconsistent

    Args:
        odour_index (int): The index for the odour response, between 0-2.
        bl (bool, optional): Should the baselined responses be used. Defaults to False.
        sb (bool, optional): Should the sniff basis responses be used. Defaults to False.
    """
    if not sb:
        data= [recs.get_binned_trial_response(f'{m}_{odour_index}',
                                              pre_trial_window=1,
                                              post_trial_window=1,
                                              baselined=bl)[1] for m in range(32)]
    else:
        # Generates the sniff basis data if not already done
        load_sniff_recs()
        data = [sb_recs.get_binned_trial_response(f'{m}_{odour_index}',
                                                   pre_trial_window=3,
                                                   post_trial_window=3,
                                                   baselined=False,
                                                   bin_size=0.05)[1] for m in range(32)]
    ## Confusingly done, but essentially swaps the axis of the data so that the mean can be
    ## calculated easily. As the number of repeats vary between the experiments, np.mean(axis=x) 
    ## cannot be used.
    data2 = [[np.array(data[i][j]).swapaxes(0, 1) for j in range(len(data[i]))] for i in range(32)]
    data3= [[i for j in k for i in j] for k in data2]
    if not sb:
        if bl:
            mean_resps_bl[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])
        else:
            mean_resps[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])
    else:
        mean_resps_sb[odour_index] = np.array([[np.mean(cell, axis=0) for cell in trial ]for trial in data3])

def return_mean_resp(odour: int, bl=False, sb=False):
    """Returns the mean response of all cells to all repeats of each stimulus type
       Tensor is pattern_type x cell x time_points

    Args:
        odour (int): The odour name (not the same as odour index). Can be 1, 3, 5
        bl (bool, optional): Use baseline subtracted data. Defaults to False.
        sb (bool, optional): Use sniff basis data. Defaults to False.

    Raises:
        ValueError: If the odour passed is not one of the three accepted values
                    throws a value error

    Returns:
        mean_resp (numpy.array): The mean_responses of all cells in tensor form
    """
    if not bl:
        if not sb:
            if odour in mean_resps:
                if mean_resps[odour] is None:
                    print(f'Generating mean odour {odour} response' % odour)
                    _generate_mean_resps(odour, bl=bl)
                return mean_resps[odour]
            else:
                raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')
        else:
            if odour in mean_resps_sb:
                if mean_resps_sb[odour] is None:
                    print(f'Generating mean odour {odour} sniff basis response' % odour)
                    _generate_mean_resps(odour, bl=bl, sb=sb)
                return mean_resps_sb[odour]
            else:
                raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')
    else:
        if sb:
            print('No sniff basis basline subtracted sorry.')
        if odour in mean_resps_bl:
            if mean_resps_bl[odour] is None:
                print(f'Generating mean odour {odour} baslined response' % odour)
                _generate_mean_resps(odour, bl=bl)
            return mean_resps_bl[odour]
        else:
            raise ValueError('Odour index does not exist, odour indexes are 1, 3, 5')

def response_plot(mean_responses: np.array,
                  cell_index: int,
                  return_shifts: bool = False,
                  pred_funcs=None,
                  return_argmaxsum: bool =False,
                  sharey: bool =False,
                  return_axis=False,
                  plot_argmaxsum=True,
                  pred_func_labels=None):
    """Generates a response plot for a single cell, showing the average response
       of a cell to all the different temporal patterns presented

    Args:
        mean_responses (np.array): The mean responses for all cells, output of return_mean_resp
        cell_index (int): Index of the cell in the array
        return_shifts (bool, optional): Should the values used to shift the responses be returned.
                                        Defaults to False.
        pred_funcs (_type_, optional): Functions to use to predict the firing. Defaults to None.
        return_argmaxsum (bool, optional): Return the sum at the argmax. Defaults to False.
        sharey (bool, optional):Should the true firing and predicted share a y axis.
                                Defaults to False.
        return_axis (bool, optional): Return the axis object. Defaults to False.
        plot_argmaxsum (bool, optional): Plot the argmaxsum. Defaults to True.
        pred_func_labels (_type_, optional): Include labels for the predicted functions.
                                             Defaults to None.

    Returns:
        array: An array containing sub arrays determined if the return_x arguments
               are set to true or not
    """
    unit = mean_responses[:, cell_index]
    _, ax = plt.subplots(1, 2, sharey=False, figsize=(12, 4))
    arg_maxes_sum: list = []
    shifts = np.array([[np.argmax(correlate(unit[i][100:], unit[j][100:]))
                        for i in range(32)]
                        for j in range(32)])
    avg_shifts = [np.mean(shifts[:, i]) - 111 for i in range(32)]
    for i in range(32):
        ax[0].plot(np.arange(-1000-avg_shifts[i]*10, 1120-10*avg_shifts[i], 10),
                   unit[i],
                   color=cm.plasma(i/32))
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
                label=f'pred_func {index}'
            else:
                label=pred_func_labels[index]
            ax2.plot(pred_func, 'o-', label=label, color=f'{index+1}')
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

def dendo_and_heatmap(bin_weights: np.array, fig:plt.figure = None, return_fig: bool = False):
    """Generates a clustered dendogram and heatmap of bin weights

    Args:
        bin_weights (np.array): Array containing cells x bin_weightings
        fig (plt.figure, optional): Figure to plot onto. Defaults to None.
        return_fig (bool, optional): Return the figure object. Defaults to False.

    Returns:
        fig (plt.figure, optional): If return_fig is set to true, it returns the figure object
    """

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
    im = ax2.imshow(bin_weights[d['leaves']].T,
                    aspect='auto',
                    cmap='seismic',
                    vmin=-np.max(abs(bin_weights)),
                    vmax=np.max(abs(bin_weights)))
    plt.colorbar(mappable=im, cax=cbar)
    if return_fig:
        return fig

def get_usrts(odour_index: int, **kwargs):
    """Generates the unit usrt tensors (units x stimuli type x repeats x time) for each odour

    Args:
        odour_index (int): The index of the odour to use, can be 1, 3, or 5

    Returns:
        unit_usrt (np.array): A tensor of all unit responses to a single odour type for all trials
    """
    if 'pre_trial_window' not in kwargs:
        kwargs['pre_trial_window'] = 0
    if 'post_trial_window' not in kwargs:
        kwargs['post_trial_window'] = 0.38
    if units_usrts[odour_index] is None:
        resps = [recs.get_binned_trial_response(f'{i}_{odour_index}', **kwargs)[1]
                 for i in range(32)]
        units_usrt = np.array([[repeation for recording in trial for repeation in np.rollaxis(recording, 1)] for trial in resps]).T
        units_usrts[odour_index] = units_usrt
    else:
        units_usrt = units_usrts[odour_index]
    return units_usrt

def get_variances(odour_index:int) -> list:
    """Finds the variances of all units to odour presentations of a certain odour identity

    Args:
        odour_index (int): The odour identity index, can be 1, 3, or 5

    Returns:
        odour_variances (list): The variances for all cells for a specific odour identity
    """
    if units_usrts[odour_index] is None:
        print('Calculating usrt')
    units_usrt = get_usrts(odour_index)
    all_vars = []
    for i in range(len(units_usrt)):
        mean_resps_var = [[np.sum(k) for k in j] for j in units_usrt[i]]
        all_vars.append([np.var(j) for j in mean_resps_var])

    odour_variances[odour_index] = all_vars
    return all_vars


def get_stable_resp_indexes(odour_index='all') -> np.array:
    """Returns the cells which respond to at least one repeat of all trial presentations

    Args:
        odour_index (string | int, optional): The odour index to . Defaults to 'all':str | int.

    Returns:
        stable_indexes (np.array): The indexes of cells which are deemed to be stably responsive
    """
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

def split_data_train(units_usrt: np.array, test_size: int =0.5, model_type=None):
    """Split the data and trains parallel models on the training and testing sections

    Args:
        units_usrt (np.array): The unit_usrt tensor to be split
        test_size (int, optional): Size of the test split. Defaults to 0.5.
        model_type (model, optional): Model to fit to data. Defaults to None.

    Returns:
        models_train (list): List of model objects fit to the training data
        models_test (list): List of model objects fit to the testing data.
    """
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

def subtract_pc(models_train: list,
                models_test: list,
                pc_index: int =0,
                subtract_mean: bool =True,
                normalise: bool =True,
                inv_bins_index: list =None):
    """Remove the contribution of one of the PCs from PCA

    Args:
        models_train (list): The bins used to construct the PCA
        models_test (list): The bins which are not considered when generating the PCs
        pc_index (int, optional): The index of the PC to be set to 0. Defaults to 0.
        subtract_mean (bool, optional): Remove the mean weighting. Defaults to True.
        normalise (bool, optional): Normalise the residuals to have an abs max of 1. 
                                    Defaults to True.
        inv_bins_index (list, optional): Sign of the bins. Defaults to None.

    Returns:
        pcad_inv_bins_train (np.array): The training bins with the contribution from the selected
                                        PC removed.
        pcad_inv_bins_train (np.array): The test bins with the contribution from the selcted PC
                                        removed.
    """
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
    pcad_inv_bins_train[:,  pc_index] = np.zeros(len(pcad_inv_bins_train))
    pcad_inv_bins_test[:,  pc_index] = np.zeros(len(pcad_inv_bins_train))
    if subtract_mean:
        pcad_inv_bins_train = pca_split.inverse_transform(pcad_inv_bins_train) - pca_split.mean_
        pcad_inv_bins_test = pca_split.inverse_transform(pcad_inv_bins_test) - pca_split.mean_
    if normalise:
        pcad_inv_bins_train = pcad_inv_bins_train/ np.max(np.abs(pcad_inv_bins_train), axis=-1)[:, np.newaxis]
        pcad_inv_bins_test = pcad_inv_bins_test/ np.max(np.abs(pcad_inv_bins_test), axis=-1)[:, np.newaxis]
    
    return pcad_inv_bins_train, pcad_inv_bins_test

def cluster_and_score_train_test(train_bins:list, test_bins:list, cluster_type = None, **kwargs):
    """Clusters cells using training and testing bin fits, and compares the silhoutte scores
        and rand indexes to determine the stabililty of the clustering, and the number of clusters
        present

    Args:
        train_bins (list): Training bin fits
        test_bins (list): Testing bin fits
        cluster_type (object, optional): Clustering type to use. Defaults to AgglomerativeClustering
    """
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