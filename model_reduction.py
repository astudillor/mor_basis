#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import util_la
import scipy.linalg as spl

def transfer_function(A, B, C, D=0.0, freq=[0.0]):

    In = util_la.eye_from_matrix(A)

    def tf(w):
        return np.abs(np.dot(C, np.linalg.solve((w*1j * In) - A, B)))

    if isinstance(freq, np.floating):
        return tf(freq)

    return np.array([tf(w) for w in freq])



def get_projected_system(A, B, C, V, W):
    Ahat = (A.T.dot(W)).T.dot(V)
    Bhat = W.T.dot(B)
    Chat = C.dot(V)
    return Ahat, Bhat, Chat

def get_modal_truncation_full(A, k):
    ''' modal truncation inefficient implementation'''

    evalues, evectors = spl.eig(A)

    func = lambda x: np.abs(x)
    index = np.argsort(func(evalues))
    evalues = evalues[index]
    evectors = evectors[:, index]
    evectors_inv = spl.inv(evectors)
    evectors_inv = util_la.biorth(evectors_inv, evectors)

    return evectors[:, :k], evectors_inv[:, :k]


def arnoldi(matvec, n, m, v, inner=np.inner,
            norm=np.linalg.norm, dtype=np.float64):
    ''' Implementation of the Arnoldi method

    References
    ----------
    .. [1] Arnoldi, W. E. The principle of minimized iterations in
           the solution of the matrix eigenvalue problem.
           Quart. Appl. Math., 1951, 9, 17-29.

    '''
    V = np.zeros(shape=(n, m + 1), dtype=dtype)
    H = np.zeros(shape=(m + 1, m), dtype=dtype)
    V[:, 0] = v / norm(v)

    for j in range(0, m):
        w = matvec(V[:, j])
        for i in range(0, j + 1):
            H[i, j] = inner(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = norm(w)
        V[:, j + 1] = w / H[j + 1, j]

    return V, H

def create_multi_shift_basis(A, B, C, sigmas):
    n = A.shape[0]
    k = len(sigmas)
    V = np.zeros((n, k))
    W = np.zeros((n, k))
    In = util_la.eye_from_matrix(A)

    for i, sigma in enumerate(sigmas):

        lu, piv = spl.lu_factor(sigma * In - A)
        V[:, i] = spl.lu_solve((lu, piv), B)

        lu, piv = spl.lu_factor(sigma * In - A.T)
        W[:, i] = spl.lu_solve((lu, piv), C)

    W = util_la.biorth(W, V)
    return V, W

def irka(A, B, C, maxit=3, sigmas=np.array([0.0])):

    old_sigmas = np.zeros(sigmas.shape)
    for j in range(0, maxit):
        V, W = create_multi_shift_basis(A, B, C, sigmas)
        Ahat, _, _ = get_projected_system(A, B, C, V, W)
        old_sigmas = np.array(sigmas)
        sigmas, _ = spl.eig(Ahat)
        sigmas = -np.sort(sigmas)
        if np.linalg.norm(sigmas - old_sigmas) < 1e-1:
            break

    return V, W
