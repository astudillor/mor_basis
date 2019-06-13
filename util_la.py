#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sparse
import scipy.sparse.linalg as spLA


def biorth(W, V):
    return (np.linalg.solve(W.T.dot(V), W.T)).T


def are_any_sparse(A, B):
    return sparse.issparse(A) or sparse.issparse(B)


def cast_matrices(A, B):
    if are_any_sparse(A, B):
        return sparse.csc_matrix(A), sparse.csc_matrix(B)
    return A, B


def solve(A, b):
    if sparse.issparse(A):
        return spLA.spsolve(A, b)
    return spl.solve(A, b)


def eye(n, m=None, issparse=False, spformat='dia'):
    if issparse:
        return sparse.eye(n, m, format=spformat)
    return np.eye(n, n)


def eye_from_matrix(A):
    n, m = A.shape
    spformat = None
    if sparse.issparse(A):
        spformat = A.getformat()
    return eye(n, m, issparse=sparse.issparse(A), spformat=spformat)


def normalize(x, norm=np.linalg.norm):
    return x / norm(x)
