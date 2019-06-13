#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import model_reduction


def main():
    filename = 'slicot/beam.mat'
    matdata = sio.loadmat(filename)
    # Load Clamped Beam problem
    A = matdata['A']
    A = A.toarray()
    n = A.shape[0]

    B = matdata['B']
    B = B.ravel()

    C = matdata['C']
    C = C.ravel()

    # Load pre-computed transfer function
    w = matdata['w']
    w = np.ravel(w)

    mag = matdata['mag']
    mag = np.ravel(mag)

    # Obtain reduce Model
    k = 26
    V, W = model_reduction.get_modal_truncation_full(A, k)

    # Compute reducing basis V and W

    # Compute the reduce model
    Ahat, Bhat, Chat = model_reduction.get_projected_system(A, B, C, V, W)

    # Compute the transfer function of the reduce model
    mag2 = model_reduction.transfer_function(Ahat, Bhat, Chat, freq=w)

    # Plotting
    fig, axarr = plt.subplots(1, 1)
    plt.loglog(w, mag, label='n = {:d} (full-model)'.format(A.shape[0]))
    plt.loglog(w, mag2, 'r--', label='k = {:d} (modal)'.format(k))
    plt.legend()
    plt.title('Bode Diagram')
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude (dB)')
    fig.suptitle('Modal truncation')
    plt.show()


if __name__ == '__main__':
    main()
