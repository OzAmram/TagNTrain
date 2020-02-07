from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ctypes

import fastjet as fj
import NsubjettinessWrapper as ns
import copy

def non_zero(arr):
    out = []
    eps = 1e-8
    for i in range(len(arr)):
        out.append(max(arr[i], eps))
    return out

def generator(filename, chunk_size=1000,total_size=1100000, start=0):

    i = 0
    i_max = total_size//chunk_size

    for i in range(i_max):
        starting = start + i*chunk_size
        stopping = start + (i+1)*chunk_size
        print("Starting ", starting, "Stopping", stopping)

        yield pd.read_hdf(filename,start=starting, stop=stopping)


R = 1.0
beta=1.0
nsub_b1 = ns.Nsubjettiness( beta, R, 0, 6 )
beta=2.0
nsub_b2 = ns.Nsubjettiness( beta, R, 0, 6 )

column_labels = ['is_signal', 'Mjj', 'Mj1', 'j1 sqrt(tau^2_1)/tau^1_1', 'j1 tau21', 'j1 tau32', 'j1 tau43', 'j1 ntrk', 
                                         'Mj2', 'j2 sqrt(tau^2_1)/tau^1_1', 'j2 tau21', 'j2 tau32', 'j2 tau43', 'j2 ntrk']
vec_size = len(column_labels)

ji_size = 3*4 + 5

has_signal_info = False

# Load files
start_evt = 000000
total_size = 400000
batch_size = 5000
iters = total_size//batch_size
fin_name = "../data/events_LHCO2020_BlackBox3.h5"
fout_name = "../data/events_train_cwbh_blackbox3.h5"



print("going to read %i events with batch size %i \n \n" % (total_size, batch_size))
l=0
for batch in generator(fin_name, chunk_size=batch_size, total_size=total_size, start =start_evt):

    print("batch %i \n" %(l))

    events_combined = batch.T





    out_vec = np.zeros((batch_size, vec_size), dtype=np.float)
    jet_infos = np.zeros((batch_size, ji_size), dtype=np.float)
    for i in range(batch_size):

        #index for pd dataframe based on overall row index, so we have to correct for the batches
        idx = l*batch_size + i  + start_evt

        if(has_signal_info):
            issignal = events_combined[idx][2100]
        else:
            issignal = 0
                
        pjs = []
        for j in range(700):
            if (events_combined[idx][j*3]>0):
                pT = events_combined[idx][j*3]
                eta = events_combined[idx][j*3+1]
                phi = events_combined[idx][j*3+2]
                
                pj = fj.PseudoJet()
                pj.reset_PtYPhiM(pT, eta, phi, 0.)
                pjs.append(pj)

                
        
        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)    
        # Save the two leading jets
        jets = jet_def(pjs)
        jets = [j for j in jets]


        jet1_conts = []
        jet2_conts = []
        for pj in pjs:
            if(jets[0].delta_R(pj) < R):
                vec = [pj.px(), pj.py(), pj.pz(), pj.e()]
                jet1_conts.extend(vec)
            if(jets[1].delta_R(pj) < R):
                vec = [pj.px(), pj.py(), pj.pz(), pj.e()]
                jet2_conts.extend(vec)

        max_tau = 4
        tau1_b1 = non_zero(nsub_b1.getTau(max_tau, jet1_conts))
        tau2_b1 = non_zero(nsub_b1.getTau(max_tau, jet2_conts))

        tau1_b2 = non_zero(nsub_b2.getTau(2, jet1_conts))
        tau2_b2 = non_zero(nsub_b2.getTau(2, jet2_conts))

        njets = len(jets)
        mjj = (jets[0] + jets[1]).m()
        if(njets > 2):
            mjjj = (jets[0] + jets[1] +jets[2]).m()
            jet3 = jets[2]
        else:
            mjjj = mjj
            jet3 = jets[1]
        met = fj.PseudoJet()
        for j in range(len(jets)):
            met += jets[j]
        met_pt = met.pt()
        met_phi = met.phi()
        obs1 = [jets[0].m(), np.sqrt(tau1_b2[0])/tau1_b1[0], tau1_b1[1]/tau1_b1[0], tau1_b1[2]/tau1_b1[1], tau1_b1[3]/tau1_b1[2], len(jet1_conts)]
        obs2 = [jets[1].m(), np.sqrt(tau2_b2[0])/tau2_b1[0], tau2_b1[1]/tau2_b1[0], tau2_b1[2]/tau2_b1[1], tau2_b1[3]/tau2_b1[2], len(jet2_conts)]
    

        jet_info = [jets[0].pt(), jets[0].eta(), jets[0].phi(), jets[0].m(), 
                    jets[1].pt(), jets[1].eta(), jets[1].phi(), jets[1].m(), 
                    jet3.pt(), jet3.eta(), jet3.phi(), jet3.m(), 
                    mjj, mjjj, met_pt, met_phi, njets]

        vec = [issignal, mjj]
        #put more massive jet first
        if(jets[0].m() > jets[1].m()):
            vec.extend(obs1)
            vec.extend(obs2)
        else:
            vec.extend(obs2)
            vec.extend(obs1)
        out_vec[i] = vec
        jet_infos[i] = jet_info
        #print(i, out_vec[i])

    if (l==0):
        #df.to_hdf(fout_name, "data", mode='w', format='t')
        #df_jet_info.to_hdf(fout_name, "jet_info", mode='w', format='t')
        with h5py.File(fout_name, "w") as f:
            f.create_dataset("data", data=out_vec, chunks = True, maxshape=(None, out_vec.shape[1]))
            f.create_dataset("jet_infos", data=jet_infos, chunks = True, maxshape=(None, jet_infos.shape[1]))
    else:
        #df.to_hdf(fout_name, "data", mode='r+', format='t', append=True)
        #df_jet_info.to_hdf(fout_name, "jet_info", mode='r+', format='t', append=True)
        with h5py.File(fout_name, "a") as f:

            f['data'].resize((f['data'].shape[0] + out_vec.shape[0]), axis=0)
            f['data'][-out_vec.shape[0]:] = out_vec

            f['jet_infos'].resize((f['jet_infos'].shape[0] + jet_infos.shape[0]), axis=0)
            f['jet_infos'][-jet_infos.shape[0]:] = jet_infos

    l+=1
    
    #fig2 = plt.figure()

    #ax = fig2.add_subplot(1, 1, 1)
 
    #sig_events = (out_vec[:,0] == 1).reshape(-1)
    #bkg_events = (out_vec[:,0] == 0).reshape(-1)

    #bkg_taus = df.iloc[bkg_events,4]
    #sig_taus = df.iloc[sig_events,4]



    #n,b,p = plt.hist(  bkg_taus, bins=50, facecolor='b', alpha=0.2,label='background')
    #plt.hist(sig_taus, bins=50, facecolor='r', alpha=0.2,label='signal')
    #plt.xlabel(r'leading jet $\tau_{21}$')
    #plt.ylabel('Number of events')
    #plt.legend(loc='upper right')
    #plt.show()
    #plt.savefig("tau21a.png")
    #exit(1)

print("Finished all batches! Output file saved to %s" %(fout_name))


            
                              


