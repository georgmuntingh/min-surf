import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.plot import Enneper_mesh, symmetry_element_meshes, bridge_meshes


def get_colors():
    c_surfs = tuple([tuple(list(mpl.colors.to_rgb(col))[:3] + [0.8]) for col in ['tab:blue', 'cornflowerblue']])
    c_plane, c_line, c_point = [tuple(list(mpl.colors.to_rgb(col))[:3] + [0.2])
                                for col in ["indianred", "forestgreen", "goldenrod"]]

    return c_surfs, c_plane, c_line, c_point


def demo(ks=(1, 2, 3, 4), N=20, azimuths=(0, 20), elevations=(90, 30), colors=get_colors(), verbose=True, savefig=False,
         showfig=True, elements=True):

    c_surfs, c_plane, c_line, c_point = colors
    for k in ks:
        if verbose:
            print(f"Enneper surface of order k={k}")

        # Collect meshes of the Enneper surface, its symmetry planes, symmetry axes, and symmetry point
        X, Y, Z, C = Enneper_mesh(k=k, N=N, colors=c_surfs)

        if elements:
            u_plane = max(np.max(X), np.max(Y))
            u_line = 0.6*np.max(np.sqrt(X**2 + Y**2))
            u_zfactor = np.max(Z) / u_plane

            Xs, Ys, Zs, Cs = symmetry_element_meshes(k, N, u_plane, u_line, u_zfactor, c_plane, c_line, c_point)
            X, Y, Z, C = bridge_meshes([X] + Xs, [Y] + Ys, [Z] + Zs, [C] + Cs)

        # We plot the surface from various view points
        for azimuth, elevation in zip(azimuths, elevations):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.view_init(elev=elevation, azim=azimuth)

            ax.plot_surface(X, Y, Z, antialiased=True, rstride=1, cstride=1, facecolors=C,
                            shade=True, linewidth=1)

            ax.set_xlim(np.min(X), np.max(X))
            ax.set_ylim(np.min(Y), np.max(Y))
            ax.set_zlim(np.min(Z), np.max(Z))
            plt.axis('off')

            if savefig:
                if elements:
                    fname = f"figs/Enneper-k-{k}-elements-elevation-{elevation}-azimuth-{azimuth}-N-{N}.png"
                else:
                    fname = f"figs/Enneper-k-{k}-elevation-{elevation}-azimuth-{azimuth}-N-{N}.png"

                plt.savefig(fname, bbox_inches='tight', dpi=300)

            if showfig:
                plt.show()


if __name__ == '__main__':
    demo()
