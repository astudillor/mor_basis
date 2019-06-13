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
import scipy.linalg as spl


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
    k = 7

    sigma = 1e-1
    In = util_la.eye_from_matrix(A)
    lu, piv = spl.lu_factor(sigma * In - A)

    # Petrov-Galerkin method
    matvec = lambda x: spl.lu_solve((lu, piv), x)
    v = util_la.normalize(B)
    V, _ = model_reduction.arnoldi(matvec, n, k, v)

    matvec2 = lambda x: spl.lu_solve((lu, piv), x, trans=1)
    w0 = util_la.normalize(C)
    W, _ = model_reduction.arnoldi(matvec2, n, k, w0)

    W = util_la.biorth(W, V)
    Ahat, Bhat, Chat = model_reduction.get_projected_system(A, B, C, V, W)
    mag2 = model_reduction.transfer_function(Ahat, Bhat, Chat, freq=w)

    sigmas = np.linspace(1e0, 1e1, 2)
    V, W = model_reduction.irka(A, B, C, maxit=3, sigmas=sigmas)
    Ahat, Bhat, Chat = model_reduction.get_projected_system(A, B, C, V, W)
    mag3 = model_reduction.transfer_function(Ahat, Bhat, Chat, freq=w)

    # Plotting
    fig, axarr = plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.loglog(w, mag, label='n = {:d} (full-model)'.format(A.shape[0]))
    plt.loglog(w, mag2, 'r--', label='k = {:d} (Petrov-Galerkin)'.format(k))
    plt.legend()
    plt.title('Bode Diagram')
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude (dB)')

    plt.subplot(1, 2, 2)
    plt.loglog(w, mag, label='n = {:d} (full-model)'.format(A.shape[0]))
    plt.loglog(w, mag3, 'r--', label='k = {:d} (IRKA)'.format(len(sigmas)))
    plt.legend()
    plt.title('Bode Diagram')
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude (dB)')
    fig.suptitle('Moment matching and IRKA')
    plt.show()


if __name__ == '__main__':
    main()
