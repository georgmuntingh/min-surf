import numpy as np
import sympy as sp

from utils.linalg import dihedral_representation, signature, nullspace
from utils.minsurf import Enneper_parametrization


def bridge_meshes(Xs, Ys, Zs, Cs):
    """
    Concatenate multiple meshes, with hidden transparent bridges, to a single mesh, so that plt.plot_surface
    uses correct drawing order between meshes (as it really should)
    :param list Xs: list of x-coordinates for each mesh
    :param list Ys: list of y-coordinates for each mesh
    :param list Zs: list of z-coordinates for each mesh
    :param list Cs: list of colors for each mesh
    :return: Concatenated meshes X_full, Y_full, Z_full, C_full
    """

    assert len(Xs) == len(Ys) == len(Zs) == len(Cs)

    if len(Xs) > 2:
        X1, Y1, Z1, C1 = bridge_meshes(Xs[1:], Ys[1:], Zs[1:], Cs[1:])
    elif len(Xs) == 2:
        X1, Y1, Z1, C1 = Xs[1], Ys[1], Zs[1], Cs[1]
    else:
        raise Exception

    X0, Y0, Z0, C0 = Xs[0], Ys[0], Zs[0], Cs[0]

    X_bridge = np.vstack(np.linspace(X0[-1, :], X1[-1, :], 1))
    Y_bridge = np.vstack(np.linspace(Y0[-1, :], Y1[-1, :], 1))
    Z_bridge = np.vstack(np.linspace(Z0[-1, :], Z1[-1, :], 1))
    color_bridge = np.empty_like(Z_bridge, dtype=object)
    color_bridge.fill((1, 1, 1, 0))  # Make the bridge transparant

    # Join surfaces
    X_full = np.vstack([X0, X_bridge, X1])
    Y_full = np.vstack([Y0, Y_bridge, Y1])
    Z_full = np.vstack([Z0, Z_bridge, Z1])
    color_full = np.vstack([C0, color_bridge, C1])

    return X_full, Y_full, Z_full, color_full


def symmetry_element_meshes(k, N, u_plane, u_line, u_zfactor, c_plane, c_line, c_point, rho=0.1, verbose=True):
    """
    :param int k: Order of the Enneper surface
    :param int N: Mesh resolution
    :param float u_plane: Mesh bound
    :param float u_line: Mesh bound
    :param float u_zfactor: Mesh bound
    :param tuple c_line: RGBA color of the lines
    :param tuple c_plane: RGBA color of the planes
    :param float rho: Radius of sphere representing symmetry point
    :param bool verbose: Whether to print auxilliary information
    :return: tuple of mesh x-, y-, z- coordinates and colors
    """
    Xs, Ys, Zs, Cs = [], [], [], []

    # Symmetry planes
    S_plane, T_plane = np.meshgrid(np.linspace(-u_plane, u_plane, N), np.linspace(-u_plane, u_plane, N))
    U_plane = np.vstack([S_plane.reshape(1, N, N), T_plane.reshape(1, N, N)])

    C_plane = np.empty((N, N), dtype=object)
    C_plane.fill(c_plane)

    # Symmetry axes
    S_line, T_line = np.meshgrid(np.linspace(-u_line, u_line, N), np.linspace(-u_line, u_line, N))
    U_line = np.vstack([S_line.reshape(1, N, N), T_line.reshape(1, N, N)])

    C_line = np.empty((N, N), dtype=object)
    C_line.fill(c_line)

    # Symmetry point
    sphere = lambda u, v: (rho*np.sin(u)*np.cos(v), rho*np.sin(u)*np.sin(v), rho*np.cos(u)+0*v)  # 0*v -> Correct dims.
    U_point = np.linspace(0, 2*np.pi, N)
    V_point = np.linspace(0, np.pi, N)
    X_point, Y_point, Z_point = sphere(U_point.reshape((-1, 1)), V_point)
    C_point = np.empty((N, N), dtype=object)
    C_point.fill(c_point)

    for n in [0, 1]:
        for m in range(2 * k + 2):
            M0 = dihedral_representation(k, m, n)
            sgns = signature(M0)
            A = M0 - np.eye(3)
            K = nullspace(A)

            if len(K) != 0:
                # Squeeze the vertical direction to align with the height of the Enneper surface
                K[2, :] = K[2, :] * u_zfactor

                if K.shape[1] == 2:
                    if verbose:
                        print("\nFound a symmetry plane, spanned by the columns of:\n", np.round(K, 8))

                    X1, Y1, Z1 = np.einsum('ji,ikl->jkl', K, U_plane)
                    Xs.append(X1)
                    Ys.append(Y1)
                    Zs.append(Z1)
                    Cs.append(C_plane)
                elif K.shape[1] == 1:
                    if verbose:
                        print("\nFound a symmetry axis, spanned by the column of:\n", np.round(K, 8))

                    X1, Y1, Z1 = np.einsum('ji,ikl->jkl', K, U_line)
                    Xs.append(X1)
                    Ys.append(Y1)
                    Zs.append(Z1)
                    Cs.append(C_line)
                elif sgns.count(-1.0) == 3:
                    if verbose:
                        print("\nFound a symmetry point")

                    Xs.append(X_point)
                    Ys.append(Y_point)
                    Zs.append(Z_point)
                    Cs.append(C_point)

    return Xs, Ys, Zs, Cs


def Enneper_mesh(k=1, u0=0.0, u1=None, v0=0, v1=2*np.pi, colors=('tab:blue', 'cornflowerblue'), N=20, verbose=True):
    """
    :param int k:
    :param float u0:
    :param float u1:
    :param float v0:
    :param float v1:
    :param tuple colors:
    :param int N:
    :return: a mesh of the Enneper surface of order k
    """
    z = sp.Symbol('z')

    if u1 is None:
        # Coefficients taken from:
        # https://minimalsurfaces.blog/home/repository/symmetrizations/higher-order-enneper-surfaces/
        coeffs = [[1, -3], [1, -4], [1, 2, -27], [1, -6, -51, 256], [1, 4, -126, -304, 3125],
                  [1, -8, -198, 1388, 8429, -46656]]
        rs = [1.73205, 1.41421, 1.274778, 1.2039268, 1.161687, 1.133786]
        # Rs = [2.1, 2, 1.5, 1.3]

        # Find the radius of maximal embedding
        u1 = np.roots(coeffs[k - 1])[-1] ** (1 / (2 * k))  # u1 = rs[k-1]
        u1 = 0.9 * u1  # Only take 90% of the maximal radius, for a better visual

    U = np.linspace(u0, u1, N)
    V = np.linspace(v0, v1, N)
    s = Enneper_parametrization(k, z)

    X, Y, Z = s(U.reshape((-1, 1)), V)

    # Enneper surface color pattern
    C = np.empty(X.shape, dtype=object)
    for v in range(N):
        for u in range(N):
            C[u, v] = colors[(u // 2) % len(colors)]

    return X, Y, Z, C
