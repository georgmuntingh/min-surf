import numpy as np
import sympy as sp
from utils.minsurf import Weierstrass_Enneper
from sympy.assumptions import assuming, Q
sp.init_printing(use_unicode=True)


def T(n, x=None):
    """
    :param n: Degree of the Chebyshev polynomial
    :param x: Variable/number
    :return: Chebyshev polynomial of the first kind of degree n in the variable/number x.
    """
    if x is None:
        x = sp.Symbol('x')

    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n >= 2:
        return sp.expand(2*x*T(n-1, x=x) - T(n-2, x=x))


def S(k):
    r, s, t = sp.symbols("r s t")

    x = sp.expand(sp.expand(t - r**(2*k+1) * T(2*k + 1, x=t/r)/(2*k + 1)).subs(r, sp.sqrt(s**2 + t**2)))
    y = sp.expand(sp.expand(-s - (-1)**k * r**(2*k+1) * T(2*k + 1, x=s/r)/(2*k + 1)).subs(r, sp.sqrt(s**2 + t**2)))
    z = sp.expand(sp.expand(2*r**(k+1) * T(k+1, x=t/r)/(k+1)).subs(r, sp.sqrt(s**2 + t**2)))

    return x, y, z


def binomial(n, k):
    assert(0 <= k <= n)

    if n == 0 or n == 1 or k == 0 or k == n:
        return 1
    else:
        return binomial(n-1, k-1) + binomial(n-1, k)


r, s, t = sp.symbols("r s t", real=True)
z = sp.Symbol('z', nonzero=True)
k = sp.Symbol('k', positive=True)
m = sp.Symbol('m', positive=True)

"""

print("Proposition 7\n")
f = 2
g = z**k
psi = Weierstrass_Enneper(f, g, z, imag_unit=sp.I)
print(f"The Weierstrass form with f={f} and g={g} yields the minimal curve")
sp.pprint(psi)

print("\nProposition 8 (for the case k=1)")
Psi = [[sp.expand(expr.subs(k, 1).diff(z, k0).subs(z, 0)) for expr in psi] for k0 in [1, 2, 3]]
Psi = sp.Matrix(Psi).transpose()
A = sp.Matrix([[sp.I**m, 0, 0], [0, (-1)**m, 0], [0, 0, (-sp.I)**m]])
M = Psi*A*Psi.inv()
print("\nThe matrix Psi=")
sp.pprint(Psi)
print("\nThe matrix Psi^-1=")
sp.pprint(Psi.inv())
print("\nThe matrix A=")
sp.pprint(A)
print("\nThe matrix M=")
sp.pprint(M)
R1 = M.subs(m, 1)
print("\nThe matrix R1=")
sp.pprint(R1)

print("\nExample 1\n")
for k in range(1, 3+1):
    print(f"\nThe Enneper surface of order k={k} has parametrization:")
    sp.pprint(S(k))

print("\nRemark 2\n")
for n in range(0, 5):
    k, eps = divmod(n, 2)
    lhs1 = sp.expand(sp.expand(r**n * T(n, t/r)).subs(r, sp.sqrt(s**2 + t**2)))
    lhs2 = sp.expand(sp.expand(r**n * T(n, s/r)).subs(r, sp.sqrt(s**2 + t**2)))
    rhs1 = sum([(-1)**(k+m) * binomial(n, 2*m + eps) * s**(n - 2*m - eps) * t**(2*m + eps) for m in range(k+1)])
    rhs2 = sum([(-1)**(k+m) * binomial(n, 2*m + eps) * t**(n - 2*m - eps) * s**(2*m + eps) for m in range(k+1)])
    print(f"For n={n}, the lhs and rhs in Remark 2 match: {lhs1 == rhs1 and lhs2 == rhs2}")

print("\nRemark 4")
S = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
T = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
for n in range(2):
    for m in range(2):
        for p in range(4):
            print(f"S^{n} R1^{p} T^{m} = ")
            sp.pprint(S**n * R1**p * T**m)

"""

