import numpy as np
import sympy as sp


def T(n, x=None):
    if x is None:
        x = sp.Symbol('x')

    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n >= 2:
        return sp.expand(2*x*T(n-1, x=x) - T(n-2, x=x))


def S(k):
    r = sp.Symbol('r')
    s = sp.Symbol('s')
    t = sp.Symbol('t')
    # r = sp.sqrt(s**2 + t**2)

    x = sp.expand(sp.expand(t - r**(2*k+1) * T(2*k + 1, x=t/r)/(2*k + 1)).subs(r, sp.sqrt(s**2 + t**2)))
    y = sp.expand(sp.expand(-s - (-1)**k * r**(2*k+1) * T(2*k + 1, x=s/r)/(2*k + 1)).subs(r, sp.sqrt(s**2 + t**2)))
    z = sp.expand(sp.expand(2*r**(k+1) * T(k+1, x=t/r)/(k+1)).subs(r, sp.sqrt(s**2 + t**2)))

    return x, y, z


for k in range(1, 3+1):
    print(f"\nThe Enneper surface of order k={k} has parametrization:\n", S(k))


