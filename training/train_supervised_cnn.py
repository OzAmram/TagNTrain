import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import tensorflow.keras as keras
import h5py


fin = "../data/jet_images.h5"
plot_dir = "../plots/"
model_dir = "../models/"
model_name  = "supervised_CNN.h5"

draw_images = False

num_data = 200000


npix = 40
input_shape = (npix, npix)
val_frac = 0.1
test_frac = 0.0
num_epoch = 20
batch_size = 200

use_j1 = True
use_both = False


if(use_both):
    j_label = "jj_"
    print("Training supervised cnn on both jets! label = jj")

elif(use_j1):
    j_label = "j1_"
    print("Training supervised cnn on leading jet! label = j1")
else:
    j_label = "j2_"
    print("Training supervised cnn on sub-leading jet! label = j2")



hf_in = h5py.File(fin, "r")

if(use_both):
    j1s = hf_in['j1_images'][:num_data]
    j2s = hf_in['j2_images'][:num_data]
    images = np.stack((j1s,j2s), axis = -1)
else:
    images = hf_in[j_label+'images'][:num_data]
    images = np.expand_dims(images, axis=-1)
jet_infos = hf_in['jet_infos'][:num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info


if(draw_images):
    images = np.squeeze(np.array(images))
    signal_bits = np.array(Y)
    signal_mask = (signal_bits == 1.)
    bkg_mask = (signal_bits == 0.)
    signal_images = images[signal_mask]
    mean_signal1 = np.mean(signal_images, axis=0)

    bkg_images = images[bkg_mask]
    mean_bkg1 = np.mean(bkg_images, axis=0)

    print("Avg pixel Sums: ", np.sum(mean_signal1), np.sum(mean_bkg1) )
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(mean_signal1, cmap='gray') #, vmin=0., vmax=max_pix)
    ax1.set_title("Signal")
    ax2.imshow(mean_bkg1, cmap='gray')#, vmin =0., vmax=max_pix)
    ax2.set_title("Background")
    plt.show()


(X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(images, Y, val=val_frac, test=test_frac)

X_train, X_val, X_test = standardize(*zero_center(X_train, X_val, X_test))
print(X_train.shape)


cnn = CNN(X_train[0].shape)
cnn.summary()

myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
cnn.compile(optimizer=myoptimizer,loss='binary_crossentropy',
          metrics = [keras.metrics.AUC()],
          callbacks = [keras.callbacks.History(), early_stop]
        )

# train model
history = cnn.fit(X_train, Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          verbose=1)

# get predictions on test data
#print(Y_predict_test)

#make_roc_curve([Y_predict_test], Y_test,  save = True, fname=plot_dir+ j_label+ "supervised_roc.png")

cnn.save(model_dir+j_label+ model_name)

