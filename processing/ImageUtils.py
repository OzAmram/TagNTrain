import numpy as np
import math
import matplotlib.pyplot as plt
import ctypes
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center

(pT_i, rap_i, phi_i, pid_i) = (0, 1, 2, 3)


def ang_dist(phi1, phi2):
    dphi = phi1 - phi2
    if(dphi < -math.pi):
        dphi += 2.* math.pi
    if(dphi > math.pi):
        dphi -= 2.*math.pi
    return dphi

def raw_moment(jet, rap_order, phi_order):
    return (jet[:,pT_i] * (jet[:,rap_i] ** rap_order) * (jet[:,phi_i] ** phi_order)).sum()
    





def my_pixelate(jet, npix=40, img_width=0.8, rotate = True, norm=True, show_images=False):
    #Augmented version of efn version to add rotation and flipping of jet in pre-processing
    #rotation code based on https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/


    # the image is (img_width x img_width) in size
    pix_width = img_width / npix
    jet_image = np.zeros((npix, npix))

    # remove particles with zero pt
    jet = jet[jet[:,pT_i] > 0]

    # get pt centroid values
    #rap_avg = np.average(jet[:,rap_i], weights=jet[:,pT_i])
    #phi_avg = np.average(jet[:,phi_i], weights=jet[:,pT_i])
    pt_sum = np.sum(jet[:,pT_i])
    m10 = raw_moment(jet, 1,0)
    m01 = raw_moment(jet, 0,1)

    #center image
    rap_avg = m10 / pt_sum
    phi_avg = m01 / pt_sum
    jet[:, rap_i] -= rap_avg
    jet[:, phi_i] -= phi_avg

    if(rotate and jet.shape[0] > 2):
        coords = np.vstack([jet[:,rap_i], jet[:,phi_i]])
        cov = np.cov(coords, aweights=jet[:,pT_i])
        evals, evecs = np.linalg.eig(cov)

        e_max = np.argmax(evals)
        rap_v1, phi_v1 = evecs[:, e_max]  # Eigenvector with largest eigenvalue

        theta = np.arctan((rap_v1)/(phi_v1))
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation_mat * coords
        rap_transformed, phi_transformed = transformed_mat.A

        transformed_v1 = rotation_mat * np.vstack([rap_v1, phi_v1])
        t_rap_v1, t_phi_v1 = transformed_v1.A
    else:
        rap_transformed, phi_transformed = jet[:,rap_i], jet[:,phi_i]

    #flip so max particle is upper right
    argmax_pt = np.argmax(jet[:,pT_i])
    ptmax_rap, ptmax_phi = rap_transformed[argmax_pt], phi_transformed[argmax_pt]
    if(ptmax_rap < 0): rap_transformed *= -1.
    if(ptmax_phi < 0): phi_transformed *= -1.
    


    if(show_images):
        #make plot  showing transformation
        scale = 0.6
        plt.plot([rap_v1*-scale*2, rap_v1*scale*2],
                 [phi_v1*-scale*2, phi_v1*scale*2], color='red')
        plt.plot(jet[:, rap_i], jet[:, phi_i], 'k.')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left

        # plot the transformed blob
        plt.plot(rap_transformed, phi_transformed, 'g.')
        plt.plot([t_rap_v1*-scale*2, t_rap_v1*scale*2],
                 [t_phi_v1*-scale*2, t_phi_v1*scale*2], color='purple')
        plt.show()



    mid_pix = np.floor(npix/2)
    # transition to indices
    rap_indices = mid_pix + np.ceil(rap_transformed/pix_width - 0.5)
    phi_indices = mid_pix + np.ceil(phi_transformed/pix_width - 0.5)

    # delete elements outside of range
    mask = np.ones(jet[:,rap_i].shape).astype(bool)
    mask[rap_indices < 0] = False
    mask[phi_indices < 0] = False
    mask[rap_indices >= npix] = False
    mask[phi_indices >= npix] = False

    #print(np.mean(mask))
    rap_indices = rap_indices[mask].astype(int)
    phi_indices = phi_indices[mask].astype(int)

    # construct grayscale image
    for pt,y,phi in zip(jet[:,pT_i][mask], rap_indices, phi_indices): 
        jet_image[phi, y] += pt

    #flip so max pixel is upper right
    #pix_max = np.unravel_index(np.argmax(jet_image, axis=None), jet_image.shape)
    #if(pix_max[0] < mid_pix): jet_image = np.flip(jet_image, 0)
    #if(pix_max[1] < mid_pix): jet_image = np.flip(jet_image, 1)
    # construct two-channel image

    # L1-normalize the pt channels of the jet image
    if norm:
        normfactor = np.sum(jet_image)
        if normfactor == 0:
            raise FloatingPointError('Image had no particles!')
        else: 
            jet_image /= normfactor

    if(show_images):
        plt.imshow(np.squeeze(jet_image))
        plt.show()
    return jet_image
