from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import entropy
from .model_defs import * 
from sklearn.utils import shuffle as sk_shuffle

fig_size = (12,9)

def print_image(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print("%5.3e" % a[i,j], end=' ')
        print("\n", end='')

# standardize(*args, channels=None, copy=False, reg=10**-10)
def standardize(*args, **kwargs):
    """Normalizes each argument by the standard deviation of the pixels in 
    args[0]. The expected use case would be `standardize(X_train, X_val, 
    X_test)`.
    **Arguments**
    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same shape in all but the first axis.
    - **channels** : _int_
        - A list of which channels (assumed to be the last axis)
        to standardize. `None` is interpretted to mean every channel.
    - **copy** : _bool_
        - Whether or not to copy the input arrays before modifying them.
    - **reg** : _float_
        - Small parameter used to avoid dividing by zero. It's important
        that this be kept consistent for images used with a given model.
    **Returns**
    - _list_ 
        - A list of the now-standardized arguments.
    """

    channels = kwargs.pop('channels', [])
    copy = kwargs.pop('copy', False)
    reg = kwargs.pop('reg', 10**-10)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if channels is None: 
        channels = np.arange(args[0].shape[-1])

    print("Before:")
    print(args[0].shape)
    #print_image(np.array(args[0]))
    

    # compute stds
    print("Stds:")
    stds = np.std(args[0], axis=0) + reg
    print(stds.shape)
    print_image(stds)
    

    # copy arguments if requested
    if copy: 
        X = [np.copy(arg) for arg in args]
    else: 
        X = args

    # iterate through arguments and channels
    for x in X: 
        for chan in channels: 
            x[...,chan] /= stds[...,chan]
    print("after")
    return X

def zero_center(*args, **kwargs):
    """Subtracts the mean of arg[0] from the arguments. The expected 
    use case would be `standardize(X_train, X_val, X_test)`.
    **Arguments**
    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same shape in all but the first axis.
    - **channels** : _int_
        - A list of which channels (assumed to be the last axis)
        to zero center. `None` is interpretted to mean every channel.
    - **copy** : _bool_
        - Whether or not to copy the input arrays before modifying them.
    **Returns**
    - _list_ 
        - A list of the zero-centered arguments.
    """

    channels = kwargs.pop('channels', None)
    copy = kwargs.pop('copy', False)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if channels is None: 
        channels = np.arange(args[0].shape[-1])

    # compute mean of the first argument
    mean = np.mean(args[0], axis=0)
    print("Means:")
    print_image(mean)

    # copy arguments if requested
    if copy: 
        X = [np.copy(arg) for arg in args]
    else: 
        X = args

    # iterate through arguments and channels
    for x in X: 
        for chan in channels: 
            x[...,chan] -= mean[...,chan]

    return X


def make_selection(j1_scores, j2_scores, percentile):
# make a selection with a given efficiency using both scores (and)
    n_points = 400
    j1_threshs = [np.percentile(j1_scores,i) for i in np.arange(0., 100., 100./n_points)]
    j2_threshs = [np.percentile(j2_scores,i) for i in np.arange(0., 100., 100./n_points)]

    combined_effs = np.array([np.mean((j1_scores > j1_threshs[i]) & (j2_scores > j2_threshs[i])) for i in range(n_points)])
    cut_idx = np.argwhere(combined_effs < (100. - percentile)/100.)[0][0]
    mask = (j1_scores > j1_threshs[cut_idx]) & (j2_scores > j2_threshs[cut_idx])
    print("Cut idx %i, eff %.3e, j1_cut %.3e, j2_cut %.3e " %(cut_idx, combined_effs[cut_idx], j1_threshs[cut_idx], j2_threshs[cut_idx]))
    print(np.mean(mask))
    return mask




def print_signal_fractions(y_true, y):
    #compute true signal fraction in signal-rich region
    y_true = y_true.reshape(-1)
    y = y.reshape(-1)
    true_sigs = (y_true > 0.9 ) & (y > 0.9)
    lost_sigs = (y_true > 0.9) & (y < 0.1 )
    #print(true_sigs.shape, lost_sigs.shape, y.shape)
    sig_frac = np.mean(true_sigs) / np.mean(y)
    outside_frac = np.mean(lost_sigs)/np.mean(1-np.mean(y))
    SR_frac = np.mean(y)
    print("Signal-rich region as a fraction of total labeled events is %.4f. Sig frac in SR is %.4f \n" % (SR_frac, sig_frac))
    print("Sig frac in bkg_region is %.4f \n" %outside_frac)
    #print("Overall signal fraction is %.4f \n" %(mass_frac * frac + (1-mass_frac)*outside_frac))



def sample_split(*args, **kwargs):
    sig_region_cut = kwargs.pop('sig_cut', 0.9)
    bkg_region_cut = kwargs.pop('bkg_cut', 0.2)
    cut_var = kwargs.pop('cut_var', np.array([]))
    sig_high = kwargs.pop('sig_high', True)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    if(cut_var.size == 0):
        raise TypeError('Must supply cut_var argument!')

    #sig_high is whether signal lives at > cut value or < cut value
    if(sig_high):
        sig_cut = cut_var > sig_region_cut
        bkg_cut = cut_var < bkg_region_cut
    else:
        sig_cut = cut_var < sig_region_cut
        bkg_cut = cut_var > bkg_region_cut



    args_sig = [x[sig_cut] for x in args]
    args_bkg = [x[bkg_cut] for x in args]



    args_zipped = [np.concatenate((args_sig[i], args_bkg[i])) for i in range(len(args))]
    labels = np.concatenate((np.ones((args_sig[0].shape[0]), dtype=np.float32), np.zeros((args_bkg[0].shape[0]), dtype=np.float32)))
    
    do_shuffle = True

    if(do_shuffle):
        shuffled = sk_shuffle(*args_zipped, labels, random_state = 123)
        args_shuffled = shuffled[:-1]
        labels = shuffled[-1]
        return args_shuffled, labels

    else:
        return args_zipped, labels


#Normalize all variables to have zero mean and unit variance
#when using this function pandas will warn you that it is ambiguous whether clean_events is working 
#with a view or a copy of the original object, but you should be ok with either
# eg events = clean_events(events)
def clean_events(pd_events):
    #print(pd_events.max(axis=0))

    #first col is signal bit and 2nd is dijet mass
    feats = pd_events.iloc[:,2:]
    idxs = feats['j2 sqrt(tau^2_1)/tau^1_1'] > 10.
    feats['j2 sqrt(tau^2_1)/tau^1_1'][idxs] = 10.
    feats = feats  - feats.mean(axis=0)
    feats = feats / feats.std(axis=0)
    #print(pd_events.max(axis=0))
    pd_events.iloc[:,2:] = feats
    return pd_events


#Normalize all variables to have zero mean and unit variance
#when using this function pandas will warn you that it is ambiguous whether clean_events is working 
#with a view or a copy of the original object, but you should be ok with either
# eg events = clean_events(events)
def clean_events_v2(events):

    #first col is signal bit and 2nd is dijet mass
    feats = events[:,2:]
    #something went wrong with the j2 tau ratios, max it at 10
    idxs = feats[:,9] > 10.
    feats[idxs,9] = 10.
    feats = feats  - feats.mean(axis=0)
    feats = feats / feats.std(axis=0)
    return feats


#create a mask that removes signal events to enforce a given fraction
#removes signal from later events (should shuffle after)
def get_signal_mask(events, sig_frac):

    num_events = events.shape[0]
    cur_frac =  np.mean(events)
    keep_frac = (sig_frac/cur_frac)
    progs = np.cumsum(events)/(num_events * cur_frac)
    keep_idxs = (events.reshape(num_events) == 0) | (progs < keep_frac)
    return keep_idxs


#create a mask that removes signal events to enforce a given fraction
#Keeps signal randomly distributed but has more noise
def get_signal_mask_rand(events, sig_frac, seed=12345):

    np.random.seed(seed)
    num_events = events.shape[0]
    cur_frac =  np.mean(events)
    drop_frac = (1. - (1./cur_frac) * sig_frac)
    rands = np.random.random(num_events)
    keep_idxs = (events.reshape(num_events) == 0) | (rands > drop_frac)
    return keep_idxs

def plot_training(hist, fname =""):
    #plot trianing and validation loss

    loss = hist['loss']

    epochs = range(1, len(loss) + 1)
    colors = ['b', 'g', 'grey', 'r']
    idx=0

    plt.figure(figsize=fig_size)
    for label,loss_hist in hist.items():
        if(len(loss_hist) > 0): plt.plot(epochs, loss_hist, colors[idx], label=label)
        idx +=1
    plt.title('Training and validation loss')
    plt.yscale("log")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    if(fname != ""): plt.savefig(fname)
    #else: 
        #plt.show(block=False)


def make_roc_curve(classifiers, y_true, colors = None, logy=False, labels = None, save=False, fname=""):
    plt.figure(figsize=fig_size)
    fs = 18
    fs_leg = 16

    for idx,scores in enumerate(classifiers):

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ys = fpr
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        cut = 1e-5
        size_before = fpr.shape[0]
        tpr = tpr[fpr > cut]
        fpr = fpr[fpr > cut]
        size_after = fpr.shape[0]


        if(logy): 
            #guard against division by 0
            fpr = np.clip(fpr, 1e-8, 1.)
            ys = 1./fpr

        lbl = 'auc'
        clr = 'navy'
        if(labels != None): lbl = labels[idx]
        if(colors != None): clr = colors[idx]
        print(lbl, " ", roc_auc)
        plt.plot(tpr, ys, lw=2, color=clr, label='%s = %.3f' % (lbl, roc_auc))
        if(size_before != size_after and tpr[0] > 0.05):
        #add a point to show the ending of the roc curve
           plt.plot(tpr[0], ys[0], marker = 's', markersize = 10, color = clr) 




    plt.xlim([0, 1.0])
    plt.xlabel('Signal Efficiency', fontsize = fs)
    if(logy):
        plt.ylim([1., 1e4])
        plt.yscale('log')
        plt.ylabel('QCD Rejection Rate', fontsize = fs)
    else:
        plt.ylim([0, 1.0])
        plt.ylabel('Background Efficiency')
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    plt.legend(loc="upper right", fontsize= fs_leg)
    if(save): 
        print("Saving roc plot to %s" % fname)
        plt.savefig(fname)
    #else: 
        #plt.show(block=False)

def make_histogram(entries, labels, colors, xaxis_label, title, num_bins, normalize = False, stacked = False, save=False, h_type = 'step', 
        h_range = None, fontsize = 16, fname="", yaxis_label = ""):
    alpha = 1.
    if(stacked): 
        h_type = 'barstacked'
        alpha = 0.2
    fig = plt.figure(figsize=fig_size)
    plt.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, density = normalize, histtype=h_type)
    plt.xlabel(xaxis_label, fontsize =fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    if(yaxis_label != ""):
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize = fontsize)
    if(save): plt.savefig(fname)
    #else: plt.show(block=False)
    return fig

def make_ratio_histogram(entries, labels, colors, axis_label, title, num_bins, normalize = False, save=False, h_range = None, weights = None, fname=""):
    h_type= 'step'
    alpha = 1.
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ns, bins, patches  = ax1.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, 
            density = normalize, weights = weights, histtype=h_type)
    max_rw = 5.
    ax1.legend(loc='upper right')
    n0 = np.clip(ns[0], 1e-8, None)
    n1 = np.clip(ns[1], 1e-8, None)
    ratio =  n0/ n1
    ratio = np.clip(ratio, 1./max_rw, max_rw)
    ax2.scatter(bins[:-1], ratio, alpha=alpha)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel(axis_label)

    if(save): plt.savefig(fname)

    return bins, ratio


#taken from https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                    y=validation_targets,
                    verbose=self.verbose,
                    sample_weight=sample_weights,
                    batch_size=self.batch_size)

            print("\n")
            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics[i-1]._name
                print("%s   %.4f " % (valuename, result))
                self.history.setdefault(valuename, []).append(result)
            print("\n")


class RocCallback(tf.keras.callbacks.Callback):
    def __init__(self,training_data,validation_data, extra_label = "", freq = 1):
        self.extra_label = extra_label
        self.freq = freq
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.skip_val = self.skip_train = False
        if(np.mean(self.y_val) < 1e-5):
            print("Not enough signal in validation set, will skip auc")
            self.skip_val = True
        if(np.mean(self.y) < 1e-5):
            print("Not enough signal in train set, will skip auc")
            self.skip_train = True


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if(epoch % self.freq != 0):
            return
        roc_train = roc_val = 0.
        msg = "\r%s" % self.extra_label
        if(not self.skip_train):
            y_pred_train = self.model.predict_proba(self.x)
            roc_train = roc_auc_score(self.y, y_pred_train)
            phrase = " roc-auc_train: %s" % str(round(roc_train,4))
            msg += phrase
        if(not self.skip_val):
            y_pred_val = self.model.predict_proba(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            phrase = " roc-auc_val: %s" % str(round(roc_val,4))
            msg += phrase
        print(msg, end =100*' ' + '\n')
        #print('\r%s roc-auc_train: %s - roc-auc_val: %s' % (self.extra_label, str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
