import jax.numpy as jnp
from jax import grad
import numpy as np

def _alpha(e, M):
    """
        Solve Eq. 20
    """
    pi = np.pi
    pi2 = pi**2
    return (3. * pi2 + 1.6 * pi * (pi - jnp.abs(M)) / (1. + e)) / (pi2 - 6.)

def _d(alpha, e):
    """
    Solve Eq. 5
    """
    return 3. * (1. - e) + alpha * e

def _r(alpha, d, M, e):
    """
    Solve Eq. 10
    """
    return 3. * alpha * d * (d - 1. + e) * M + M**3

def _q(alpha, d, e, M):
    """
    Solve Eq. 9
    """
    return 2. * alpha * d * (1. - e) - M**2

def _w(r, q):
    """
    Solve Eq. 14
    """
    return (jnp.abs(r) + jnp.sqrt(q**3 + r**2))**(2. / 3.)

def _E1(d, r, w, q, M):
    """
    Solve Eq. 15
    """
    return (2. * r * w / (w**2 + w * q + q**2) + M) / d

def _f01234(e, E, M):
    """
    Solve Eq. 21, 25, 26, 27, and 28 (f, f', f'', f''', and f'''')
    """
    f0 = E - e * jnp.sin(E) - M
    f1 = 1. - e * jnp.cos(E)
    f2 = e * jnp.sin(E)
    return f0, f1, f2, 1. - f1, -f2

def _d3(E, f):
    """
    Solve Eq. 22 
    """
    return -f[0] / (f[1] - 0.5 * f[0] * f[2] / f[1])

def _d4(E, f, d3):
    """
    Solve Eq. 23
    """
    return -f[0] / (f[1] + 0.5 * d3 * f[2] + (d3**2) * f[3] / 6.)

def _d5(E, f, d4):
    """
    Solve Eq. 24
    """
    return -f[0] / (f[1] + 0.5 * d4 * f[2] + d4**2 * f[3] / 6. + d4**3 * f[4] / 24.)

def getE(M, e):
    """
    Solve Kepler's Equation for the "eccentric anomaly", E.
    Parameters
    ----------
    M : float
        Mean anomaly.
    e : float
        Eccentricity
    Returns
    -------
    Eccentric anomaly: float
        The solution of Kepler's Equation
    """
    # For the mean anomaly, use values between
    # -pi and pi.
    flip = False
    pi=np.pi
    M = M - (jnp.floor(M / (2. * pi)) * 2. * pi)
    if M > jnp.pi:
        M = 2. * jnp.pi - M
        # Flip the sign of result
        # if this happened
        flip = True
    e = e
    if M == 0.0:
        return 0.0
    alpha = _alpha(e, M)
    d = _d(alpha, e)
    r = _r(alpha, d, M, e)
    q = _q(alpha, d, e, M)
    w = _w(r, q)
    E1 = _E1(d, r, w, q, M)
    f = _f01234(e, E1, M)
    d3 = _d3(E1, f)
    d4 = _d4(E1, f, d3)
    d5 = _d5(E1, f, d4)
    # Eq. 29
    E5 = E1 + d5
    if flip:
        E5 = 2. * pi - E5
    E = E5
    return E5

if __name__=="__main__":
    M=0.1
    e=0.2
    Ec = getE(M,e)
    print(Ec)
    grad_f = grad(getE, argnums=[0, 1])
    for e in np.linspace(0,0.99,100):
        print(grad_f(M,e))
