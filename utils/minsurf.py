import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify, implemented_function


def Weierstrass_Enneper(f, g, z, imag_unit=1j):
    """
    Compute the Weierstrass Enneper parametrization for given 'Weierstrass data'.
    :param sympy expression f: h'/g, with h the height function
    :param sympy expression g: Gauss map.
    :param sympy variable z: Complex variable.
    :param imag_unit: Representation of imaginary unit, e.g. 1j (numpy) or sympy.I.
    :return: Weierstrass Enneper parametrization as a tuple of sympy expressions.
    """
    psi1 = sp.integrate(f*(1-g**2)/2, z)
    psi2 = imag_unit*sp.integrate(f*(1+g**2)/2, z)
    psi3 = sp.integrate(f*g, z)

    return psi1, psi2, psi3


def Enneper_parametrization(k, z, verbose=True):
    """
    Compute a parametrization of the higher order Enneper surface of order k.
    :param int k: Order of the Enneper surface
    :param sympy variable z: Complex variable.
    :param bool verbose: Whether to output auxilliary information.
    :return: function s parametrizing the higher order Enneper surface of order k
    """
    f = 2
    g = z**k

    psi = Weierstrass_Enneper(f, g, z)
    if verbose:
        print(f"Minimal curve psi_{k} = {psi}")

    psi_func = lambdify(z, psi)
    polar = lambda r, phi: r*np.cos(phi) + 1j*r*np.sin(phi)

    return lambda u, v: map(np.real, psi_func(polar(u, v)))
