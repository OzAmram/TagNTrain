import sys
sys.path.append('..')
from utils.TrainingUtils import *
#from rootconvert import to_root
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py

plot_dir = "../plots/"
colors = ['g', 'b']
size = 10
fs = 28
fs_leg = 24

bkg_ratio_ae = [1.10, 0.9, .85]
bkg_ratio_ae_uncs = [0.1, 0.05, 0.03]

effs = [0.2, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.0009999999999999432, 0.0004999999999999716, 0.00010000000000005117]
recip_effs = 1./np.array(effs)


bkg_ratio_ae = [1.00963624078016, 1.0123289030491327, 1.011982459425028, 1.0136081390313174, 1.0151118235024321, 1.020115719554012, 1.016609912324397, 1.0070405198383656, 0.9914944109882401, 0.9953730858124265, 1.0223788746055666, 1.0142684182629909, 1.0223881189953294, 0.98809911233938, 1.209932592300156]
bkg_ratio_ae_uncs = [0.00862260158349175, 0.010315798910471902, 0.010918868604620102, 0.011765725415879748, 0.012344261394868135, 0.013118432689108618, 0.014089098517876965, 0.015539922379880425, 0.018314745909632682, 0.01978529667157993, 0.022387638225916377, 0.02655931919339364, 0.03357513500235352, 0.03888089833302965, 0.06698788155084245]
bkg_ae_poisson_unc = [0.004347291953461809, 0.006197447672923631, 0.006946789918015513, 0.008046742493087794, 0.008837799145921673, 0.009907791231325873, 0.011487425279926293, 0.014174775185418743, 0.020153755044768934, 0.02338261867724778, 0.028759865427152538, 0.04096159602595202, 0.0646846227353151, 0.09128709291752769, 0.2]



bkg_ratio_TNT = [1.0001181431879433, 1.0046944842847376, 1.0109805510841394, 1.023879256700947, 1.022063616369001, 1.0223343559311413, 1.0165373260441304, 0.9910236360395387, 1.0120760907154949, 1.0201444280232401, 1.0073423715477974, 0.9142662918728248, 0.9439624006978966, 1.1109103236181554, 1.6359697150408452]
bkg_ratio_TNT_uncs = [0.00856110593071351, 0.010292928766401975, 0.01097726380882658, 0.011974185831578122, 0.012567439584007602, 0.01337986671755598, 0.014455004653653219, 0.015930982444213843, 0.019960647210401767, 0.021912678502759166, 0.024600266532975264, 0.028373058655274074, 0.03913039050079409, 0.05394615127795847, 0.10843875504563165]
bkg_TNT_poisson_unc =  [0.004348031575346029, 0.00624220211798653, 0.0070337124880089944, 0.008211630979566228, 0.00906957197482501, 0.010276021647901047, 0.012096577515622115, 0.015264062771816588, 0.0232181730106286, 0.02765006318046655, 0.03551104121142175, 0.0546358364708153, 0.09901475429766743, 0.14744195615489714, 0.3333333333333333]





plt.figure(figsize=fig_size)
fig, ax = plt.subplots(figsize = fig_size)
ax.plot(effs, bkg_ratio_ae,  c = colors[0],  label = "Autoencoder")
ax.plot(effs, bkg_ratio_TNT, c = colors[1],  label = "TNT")

bkg_ratio_ae = np.array(bkg_ratio_ae)
bkg_ae_poisson_unc = np.array(bkg_ae_poisson_unc)
bkg_ratio_TNT = np.array(bkg_ratio_TNT)
bkg_TNT_poisson_unc = np.array(bkg_TNT_poisson_unc)

#error bands
alpha = 0.5
ae_low = bkg_ratio_ae - bkg_ae_poisson_unc
ae_high = bkg_ratio_ae + bkg_ae_poisson_unc
TNT_low = bkg_ratio_TNT - bkg_TNT_poisson_unc
TNT_high = bkg_ratio_TNT + bkg_TNT_poisson_unc

ax.fill_between(effs, ae_low, ae_high, facecolor = colors[0], alpha = alpha, interpolate = True)
ax.fill_between(effs, TNT_low, TNT_high, facecolor = colors[1], alpha = alpha, interpolate = True)

one = [1.] * len(effs)
one_m = [0.95] * len(effs)
one_p = [1.05] * len(effs)
ax.plot(effs, one, linestyle = '-', color = 'black')
ax.plot(effs, one_m, linestyle = '--', color = 'black')
ax.plot(effs, one_p, linestyle = '--', color = 'black')

ax.set_xlabel('Selection Efficiency', fontsize = fs)
ax.set_ylabel('Predicted Bkg. / True Bkg.', fontsize = fs)
plt.ylim([0., 1.5])
plt.xscale('log')

plt.tick_params(axis='x', labelsize=fs_leg)
plt.tick_params(axis='y', labelsize=fs_leg)
plt.legend(loc="lower right", fontsize= fs_leg)
plt.savefig(plot_dir + "cut_and_count_bkg_check.png")
