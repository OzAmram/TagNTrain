import sys
sys.path.append('..')
from TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import tensorflow.keras as keras
import h5py



fin = "../../data/events_cwbh_v3.h5"
pd_events = pd.read_hdf(fin)

#pandas will warn us that it is ambiguous whether clean_events is working with a view or a copy of
#the original object, but we are ok with either
pd_events = clean_events(pd_events)



plot_dir = "plots_v3/"
model_dir = "models_v3/"
model_name  = "supervised_dense.h5"


num_data = 150000


val_frac = 0.1
test_frac = 0.1
num_epoch = 20
batch_size = 100

use_j1 = False


if(use_j1):
    j_label = "j1_"
    print("Training supervised dense net on leading jet! label = j1")
    tau_start = 2
    tau_end = 8
else:
    j_label = "j2_"
    print("Training supervsised dense net  on sub-leading jet! label = j2")
    tau_start = 8
    tau_end = 14



X = pd_events.iloc[:num_data, tau_start:tau_end].values
Y = pd_events.iloc[:num_data, [0]].values


(X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(X, Y, val=val_frac, test=test_frac)

print(X_train.shape)


dense = dense_net(X_train.shape[1])
dense.summary()

myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
dense.compile(optimizer=myoptimizer,loss='binary_crossentropy',
          metrics = [keras.metrics.AUC()],
          callbacks = [keras.callbacks.History(), early_stop]
        )

# train model
history = dense.fit(X_train, Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          verbose=1)

# get predictions on test data
#Y_predict_test = cnn.predict(X_test, batch_size=1000)
#print(Y_predict_test)

#make_roc_curve([Y_predict_test], Y_test,  save = True, fname=plot_dir+ j_label+ "supervised_roc.png")

dense.save(model_dir+j_label+ model_name)

