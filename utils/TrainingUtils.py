from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import entropy
from .model_defs import * 
from sklearn.utils import shuffle as sk_shuffle

fig_size = (12,9)


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
#assumes signal is randomly distributed in the events
def get_signal_mask(events, sig_frac, seed=12345):

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

    for idx,scores in enumerate(classifiers):

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ys = fpr
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



    plt.xlim([0, 1.0])
    plt.xlabel('Signal Efficiency')
    if(logy): 
        plt.ylim([1., 1e4])
        plt.yscale('log')
        plt.ylabel('QCD Rejection Rate')
    else: 
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
        plt.xlabel('QCD Efficiency')
        plt.ylim([0, 1.0])

    plt.legend(loc="lower right")
    if(save): 
        print("Saving roc plot to %s" % fname)
        plt.savefig(fname)
    #else: 
        #plt.show(block=False)

def make_histogram(entries, labels, colors, axis_label, title, num_bins, normalize = False, stacked = False, save=False, h_type = 'step', h_range = None, fname=""):
    alpha = 1.
    if(stacked): 
        h_type = 'barstacked'
        alpha = 0.2
    fig = plt.figure(figsize=fig_size)
    plt.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, density = normalize, histtype=h_type)
    plt.xlabel(axis_label)
    plt.title(title)
    plt.legend(loc='upper right')
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
