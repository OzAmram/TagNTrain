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
model_name = "supervised_CNN.h5"
fout_name = 'supervised_03p.h5'
model_type = 3



filt_sig = True
sig_frac = 0.003

num_data = 300000
data_start = 800000

percentile_cut = 99.5

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
    mask = make_selection(j1_scores, j2_scores, percentile_cut)

else:
    jj_model = load_model(model_dir + "jj_" + model_name)
    X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
    X = standardize(*zero_center(X))[0]
    jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)
    thresh = np.percentile(jj_scores, percentile_cut)
    mask =  jj_scores > thresh


newdf = pd.DataFrame()
newdf['is_signal'] = Y[mask]
newdf['Mjj'] = dijet_mass[mask]


print(Y[mask].shape) 
h5File = h5py.File(fout_name,'w')
h5File.create_dataset('test', data=newdf.values,  compression='lzf')
h5File.close()
