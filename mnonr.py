""" A Generator of Multivariate Non-Normal Random Numbers

    Reference:
        A method of generating multivariate non-normal random numbers
        with desired multivariate skewness and kurtosis

        https://doi.org/10.3758/s13428-019-01291-5

    Code modified from:
        https://github.com/wqu-nd/mnonr

"""

import numpy as np
import scipy.optimize as optim


def multouni(d, ms, mk):
    beta1 = ms
    beta2 = mk

    # Fourth moment of ksi
    f = beta2 / d - (d - 1)

    # third moment of ksi
    t = np.sqrt(beta1 / d)

    return [t, f]


def fleishtarget(x, a):
    b, c, d = x[0], x[1], x[2]
    g1, g2 = a[0], a[1]

    loss = (
        (1 - ((b ** 2) + 2 * (c ** 2) + 6 * b * d + 15 * (d ** 2))) ** 2
        + (g1 - (6 * (b ** 2) * c + 8 * (c ** 3) + 72 * b * c * d + 270 * c * (d ** 2)))
        ** 2
        + (
            g2
            - (
                3 * (b ** 4)
                + 60 * (b ** 2) * (c ** 2)
                + 60 * (c ** 4)
                + 60 * (b ** 3) * d
                + 936 * b * (c ** 2) * d
                + 630 * (b ** 2) * (d ** 2)
                + 4500 * (c ** 2) * (d ** 2)
                + 3780 * b * (d ** 3)
                + 10395 * (d ** 4)
            )
        )
        ** 2
    )

    return loss


def reltol(xk):
    tol = 1e-10
    val = fleishtarget(xk, reltol.a)

    if not hasattr(reltol, "last"):
        reltol.last = val
        return False

    if reltol.last - val < (tol * abs(val) + tol):
        return True

    reltol.last = val
    return False


def findcoef(tf, coef):
    result = optim.minimize(
        fleishtarget,
        coef,
        args=tf,
        method="BFGS",
        callback=reltol,  # TODO
        options={
            "maxiter": 1e8,
            "finite_diff_rel_step": [1e-10, 1e-10, 1e-10],
        },
    )

    return result.x


def mnonr(
    n: int,
    p: int,
    ms: float,
    mk: float,
    cov: np.ndarray,
    coef: np.ndarray = np.array([0.9, 0.4, 0]),
) -> np.ndarray:
    """Multivariate Non-normal Random Number Generator based on Multivariate Measures

    Args:
        n (int): Sample size
        p (int): Number of variables
        ms (float): Value of multivariate skewness
        mk (float): Value of multivariate kurtosis
        cov (np.ndarray): Covariance matrix
        coef (np.ndarray, optional): Vector with 3 numbers for initial polynominal coefficients' (b,c,d). Defaults to (0.9,0.4,0).

    Return:
        Multivariate data matrix
    """

    if cov.shape != (p, p):
        raise ValueError("Incompatible covariance matrix.")
    if (cov != cov.T).any():
        raise ValueError("Covariance matrix must be symmetric.")

    ev, _ = np.linalg.eig(cov)

    if not (ev >= -1e-6 * np.abs(ev[0])).all():
        raise ValueError("Covariance matrix is not positive definite.")

    # Multivariate skewness and kurtosis range
    sug_min_mk = p * (p + 0.774)
    sug_mk = 1.641 * ms + sug_min_mk
    sug_ms = (mk - sug_min_mk) / 1.641

    if ms < 0:
        raise ValueError("Multivariate skewness must be non-negtive.")

    if mk < sug_min_mk:
        raise ValueError(
            "The minimun multivariate kurtosis in your setting should be %.3f"
            % sug_min_mk
        )

    if not mk >= sug_mk:
        raise ValueError(
            "Multivariate skewness and kurtosis must follow the range of: MK>=1.641*MS+d*(d+0.774) and MS cannot be negative, where d is the number of variables.\n For your reference:\n For the given d and multivariate skewness, the kurtosis must be no less than {:.3f}.".format(
                sug_mk
            ),
            "\n For the given d and multivariate kurtosis, the skewness must be no more than {:.3f}.".format(
                sug_ms
            ),
        )

    z = np.random.normal(0, 1, (n, p))
    reltol.a = multouni(p, ms, mk)

    coef = findcoef(reltol.a, coef)

    b, c, d = coef[0], coef[1], coef[2]
    a = -1 * c
    xi = a + b * z + c * (z ** 2) + d * (z ** 3)

    x = np.zeros((n, p))
    r = np.linalg.cholesky(cov).T

    for j in range(n):
        for m in range(p):
            for i in range(p):
                x[j, m] += r[i, m] * xi[j, i]

    return x, coef


if __name__ == "__main__":
    data, coef = mnonr(
        n=5000,
        p=100,
        ms=10,
        mk=12000,
        # cov=np.eye(3) / 2 + np.ones((3, 3)) / 2,
        cov=np.eye(100),
    )

    # for i, row in enumerate(data):
    #     print("[{:2d}]".format(i), end=" ")
    #     for x in row:
    #         print("{:7.3f}".format(x), end=" ")
    #     print()

    max, min = (
        data[np.unravel_index(np.argmax(np.abs(data)), data.shape)],
        data[np.unravel_index(np.argmin(np.abs(data)), data.shape)],
    )
    print("max={:.3f} min={:.3f}".format(max, min))
    print("coef: {}".format(coef))
