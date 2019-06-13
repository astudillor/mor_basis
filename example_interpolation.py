#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import model_reduction
import util_la


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
    # Compute reducing basis V and W
    n = A.shape[0]
    k = 1

    np.random.seed(5)
    V = np.random.rand(n, k)
    W = np.random.rand(n, k)
    V[:, 0] = util_la.normalize(V[:, 0])
    W[:, 0] = util_la.normalize(W[:, 0])
    W = util_la.biorth(W, V)

    # Compute the reduce model
    Ahat, Bhat, Chat = model_reduction.get_projected_system(A, B, C, V, W)

    # Compute the transfer function of the reduce model
    mag2 = model_reduction.transfer_function(Ahat, Bhat, Chat, freq=w)

    sigma = 1e-1
    In = util_la.eye_from_matrix(A)
    V[:, 0] = np.linalg.solve((sigma * In - A), B)
    V[:, 0] = util_la.normalize(V[:, 0])
    Ahat, Bhat, Chat = model_reduction.get_projected_system(A, B, C, V, W)
    mag3 = model_reduction.transfer_function(Ahat, Bhat, Chat, freq=w)

    # Plotting
    fig, axarr = plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.loglog(w, mag, label='n = {:d} (full-model)'.format(A.shape[0]))
    plt.loglog(w, mag2, 'r--', label='k = {:d} (random basis)'.format(k))
    plt.legend()
    plt.title('Bode Diagram')
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(1, 2, 2)
    plt.loglog(w, mag, label='n = {:d} (full-model)'.format(A.shape[0]))
    plt.loglog(w, mag3, 'r--', label="k = {:d} ".format(k) + r"$(\sigma I - A)^{-1} b\in V$")
    plt.legend()
    plt.title('Bode Diagram')
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude (dB)')
    fig.suptitle('Interpolation condition')
    plt.show()


if __name__ == '__main__':
    main()
