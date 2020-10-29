import sys
sys.path.append('..')
from utils.TrainingUtils import *
#from rootconvert import to_root
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py

fin = "../data/jet_images_v3.h5"

model_dir = "../models/"

#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
model_name = "TNT1_CNN_mjj_03p.h5"
model_type = 0
#model_name = "auto_encoder_1p.h5"
#model_type = 1



filt_sig = True
sig_frac = 0.003

num_data = 300000
#data_start = 200000
data_start = 800000

#sig_percentile_cut = 99.5
cuts = [90., 95., 97., 99., 99.5, 99.9, 99.95]
#cuts = [80., 90., 92., 94., 95., 96., 97., 98., 99., 99.25, 99.5, 99.75, 99.9, 99.95, 99.99]
effs = []
poisson_uncs = []
bkg_ratios = []
bkg_ratios_unc = []

bkg_percentile_cut = 40.

use_dense = (model_type == 2 or model_type == 4)
# reading images
hf_in = h5py.File(fin, "r")

j1_images_raw = hf_in['j1_images'][data_start:data_start + num_data]
j1_images = np.expand_dims(j1_images_raw, axis=-1)
j1_images = standardize(*zero_center(j1_images))[0]
j2_images_raw = hf_in['j2_images'][data_start:data_start + num_data]
j2_images = np.expand_dims(j2_images_raw, axis=-1)
j2_images = standardize(*zero_center(j2_images))[0]

jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info
dijet_mass = jet_infos[:,9]

if(filt_sig):
    sig_mask = get_signal_mask(Y, sig_frac)
    j1_images = j1_images[sig_mask]
    j2_images = j2_images[sig_mask]
    j1_images_raw = j1_images_raw[sig_mask]
    j2_images_raw = j2_images_raw[sig_mask]
    Y = Y[sig_mask]
    dijet_mass = dijet_mass[sig_mask]

batch_size = 1000




if(model_type <= 2):
    j1_model = load_model(model_dir + "j1_" + model_name)
    j2_model = load_model(model_dir + "j2_" + model_name)
    if(model_type == 0):
        print("computing scores")
        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    elif(model_type ==1):
        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
        j1_scores =  np.mean(np.square(j1_reco_images - j1_images), axis=(1,2)).reshape(-1)
        j2_scores =  np.mean(np.square(j2_reco_images - j2_images), axis=(1,2)).reshape(-1)
    elif(model_type == 2):
        j1_scores = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_dense_inputs, batch_size = batch_size)

    j1_scores = j1_scores.reshape(-1)
    j2_scores = j2_scores.reshape(-1)

n_points = 2000
j1_threshs = [np.percentile(j1_scores,i) for i in np.arange(0., 100., 100./n_points)]
j2_threshs = [np.percentile(j2_scores,i) for i in np.arange(0., 100., 100./n_points)]

for sig_percentile_cut in cuts:
    eff = (100. - sig_percentile_cut)/100.
    print(" \n\n \n Now doing %.3f cut !" % eff)

    combined_effs = np.array([np.mean((j1_scores > j1_threshs[i]) & (j2_scores > j2_threshs[i])) for i in range(n_points)])
    sig_cut_idx = np.argwhere(combined_effs < (100. - sig_percentile_cut)/100.)[0][0]
    bkg_cut_idx = np.argwhere(combined_effs < (100. - bkg_percentile_cut)/100.)[0][0]

    j1_sig_cut = j1_threshs[sig_cut_idx]
    j2_sig_cut = j2_threshs[sig_cut_idx]
    sig_mask = (j1_scores > j1_sig_cut) & (j2_scores > j2_sig_cut)

    j1_bkg_cut = j1_threshs[bkg_cut_idx]
    j2_bkg_cut = j2_threshs[bkg_cut_idx]

    #print("Sig Cut idx %i, eff %.4f, j1_cut %.3e, j2_cut %.3e " %(sig_cut_idx, combined_effs[sig_cut_idx], j1_threshs[sig_cut_idx], j2_threshs[sig_cut_idx]))
    #print("Bkg Cut idx %i, eff %.4f, j1_cut %.3e, j2_cut %.3e " %(bkg_cut_idx, combined_effs[bkg_cut_idx], j1_threshs[bkg_cut_idx], j2_threshs[bkg_cut_idx]))

    j1_bkg_mask = j1_scores < j1_bkg_cut
    j2_bkg_mask = j2_scores < j2_bkg_cut

    est_j1_cut_eff = np.sum(j1_scores[j2_bkg_mask] > j1_sig_cut) / j1_scores[j2_bkg_mask].shape[0]
    est_j2_cut_eff = np.sum(j2_scores[j1_bkg_mask] > j2_sig_cut) / j2_scores[j1_bkg_mask].shape[0]

    est_j1_cut_eff_unc = np.sqrt(np.sum(j1_scores[j2_bkg_mask] > j1_sig_cut)) / j1_scores[j2_bkg_mask].shape[0]
    est_j2_cut_eff_unc = np.sqrt(np.sum(j2_scores[j1_bkg_mask] > j2_sig_cut)) / j2_scores[j1_bkg_mask].shape[0]

    true_j1_cut_eff = np.sum(j1_scores[Y < 0.1] > j1_sig_cut) / j1_scores[Y < 0.1].shape[0]
    true_j2_cut_eff = np.sum(j2_scores[Y < 0.1] > j2_sig_cut) / j2_scores[Y < 0.1].shape[0]

    print("J1 cut: estimated eff, true efficiency = %.4f +/- %.4f,  %.4f \n" % (est_j1_cut_eff, est_j1_cut_eff_unc, true_j1_cut_eff))
    print("J2 cut: estimated eff, true efficiency = %.4f +/- %.4f,  %.4f \n" % (est_j2_cut_eff, est_j1_cut_eff_unc, true_j2_cut_eff))

    est_bkg = est_j1_cut_eff * est_j2_cut_eff * Y.shape[0]
    est_bkg_unc = np.sqrt((est_j2_cut_eff * Y.shape[0] * est_j1_cut_eff_unc)**2 + (est_j1_cut_eff * Y.shape[0] * est_j2_cut_eff_unc)**2 )
    sys_bkg_unc = 0.03*est_bkg

    true_bkg = Y[sig_mask & (Y < 0.1) ].shape[0]
    obs_events = Y[sig_mask].shape[0]
    unc = np.sqrt(obs_events + est_bkg_unc**2 + sys_bkg_unc**2)
    significance = (obs_events - est_bkg)/unc
    expected_sig= (obs_events - true_bkg)/unc

    bkg_ratios.append(est_bkg / true_bkg)
    bkg_ratios_unc.append(est_bkg_unc /  true_bkg)
    poisson_uncs.append(np.sqrt(true_bkg)/ true_bkg)
    effs.append(eff)

    print("\n Eff %.3f: Obs %.0f Est bkg %.0f +/- %.0f, True bkg %.0f, obs sig %.2f, expected sig %.2f \n" % (eff, obs_events, est_bkg, est_bkg_unc, true_bkg, significance, expected_sig))

print(effs)
print(bkg_ratios)
print(bkg_ratios_unc)
print(poisson_uncs)
