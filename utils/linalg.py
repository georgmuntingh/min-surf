import numpy as np


def nullspace(A, atol=1e-13, rtol=0):
    """
    For given tolerances, return a matrix whose columns span the null space of a matrix A.
    :param matrix A:
    :param float atol:
    :param float rtol:
    :return np.array: matrix whose columns space the nullspace of A
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return ns


def signature(M, atol=1e-8):
    """
    Return the signature of a matrix M, in the form of a list with +1.0 for every positive eigenvalue,
    and -1.0 for every negative eigenvalue, discarding nonreal eigenvalues.
    :param matrix M:
    :return list: Signature
    """

    return [np.sign(x.real) for x in np.linalg.eigvals(M) if np.abs(x.imag) < atol]


def dihedral_representation(k, m, n):
    """
    For the dihedral group D of order 4k + 4, return the element as a matrix S^n R^m,
    with S the reflection and R an (inproper) rotation matrix.
    :param int m: Power of the (inproper) rotation matrix
    :param int n: Power of the reflection
    :param int k: Order of the Enneper surface
    :return: Element of the D with parameters (m,n)
    """
    # Dihedral group of order 4k+4
    S_math = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_math = lambda k: np.array([[np.cos(np.pi/(k+1)), np.sin(np.pi/(k+1)), 0],
                                 [-np.sin(np.pi/(k+1)), np.cos(np.pi/(k+1)), 0],
                                 [0, 0, -1]])

    return np.linalg.matrix_power(S_math, n) @ np.linalg.matrix_power(R_math(k), m)
