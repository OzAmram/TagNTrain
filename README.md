# TagNTrain


Code to reproduce results from Tag N' Train paper.

LHCO R&D Dataset used in the paper is available from here: https://zenodo.org/record/2629073

The 'processing' directory processes the data from the LHCO format into the jet
images or dense inputs. It relies on fastjet library (with the python wrapper)
to be installed. It has its own python wrapper for fastjet contrib to 
compute n-subjettiness.
Warning that the resulting file of jet images for all 1.1M events in the LHCO dataset is quite large
(25 GB)

The 'training' directory has scripts to train the autoencoders, 
Tag N' Train network, CWoLa Hunting network and supervised classifiers.

The 'plotting' directory has scripts to make all the plots in the paper (and others).
