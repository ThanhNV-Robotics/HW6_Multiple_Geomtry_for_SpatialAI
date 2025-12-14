# % Direct Image Alignment
# % Multiple View Geometry Exercise 8
# % Jakob Engel, Robert Maier

# % select dataset
# dataset = 1;
# % 1 = fr1/xyz:
# % expected result approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004
# % 2 = fr3/long_office_household:
# % expected result approximately  0.2979   -0.0106    0.0452   -0.0041   -0.0993   -0.0421

# % use numeric/analytic derivatives (true=numeric, false=analytic)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from downscale import downscale
from deriveErrNumeric import deriveErrNumeric
from deriveErrAnalytic import deriveErrAnalytic
from calcErr import se3Exp

from deriveErrNumeric import se3Log
# -----------------------------
# Direct Image Alignment
# Multiple View Geometry – Exercise 8
# Jakob Engel, Robert Maier
# -----------------------------

# select dataset
dataset = 1
# 1 = fr1/xyz
# expected result approx:
# [-0.0018, 0.0065, 0.0369, -0.0287, -0.0184, -0.0004]
#
# 2 = fr3/long_office_household
# expected result approx:
# [0.2979, -0.0106, 0.0452, -0.0041, -0.0993, -0.0421]

# use numeric/analytic derivatives
useNumeric = False  # true=numeric, false=analytic

# -----------------------------
# Load dataset
# -----------------------------
if dataset == 1:
    K = np.array([
        [517.3, 0.0, 318.6],
        [0.0, 516.5, 255.3],
        [0.0, 0.0, 1.0]
    ])

    c1 = cv2.imread("rgb/1305031102.275326.png", cv2.IMREAD_GRAYSCALE).astype(float)
    c2 = cv2.imread("rgb/1305031102.175304.png", cv2.IMREAD_GRAYSCALE).astype(float)

    d1 = cv2.imread("depth/1305031102.262886.png", cv2.IMREAD_UNCHANGED).astype(float) / 5000.0
    d2 = cv2.imread("depth/1305031102.160407.png", cv2.IMREAD_UNCHANGED).astype(float) / 5000.0

else:
    K = np.array([
        [535.4, 0.0, 320.1],
        [0.0, 539.2, 247.6],
        [0.0, 0.0, 1.0]
    ])

    c1 = cv2.imread("rgb/1341847980.722988.png", cv2.IMREAD_GRAYSCALE).astype(float)
    c2 = cv2.imread("rgb/1341847982.998783.png", cv2.IMREAD_GRAYSCALE).astype(float)

    d1 = cv2.imread("depth/1341847980.723020.png", cv2.IMREAD_UNCHANGED).astype(float) / 5000.0
    d2 = cv2.imread("depth/1341847982.998830.png", cv2.IMREAD_UNCHANGED).astype(float) / 5000.0


# -----------------------------
# Initialization
# -----------------------------
xi = np.zeros((6, 1))   # column vector like MATLAB

# -----------------------------
# Pyramid levels
# -----------------------------
for lvl in range(5, 0, -1):
    print("Level:", lvl)

    # Downscale reference frame
    IRef, DRef, Klvl = downscale(c1, d1, K, lvl)

    # Downscale target frame
    I, D, _ = downscale(c2, d2, K, lvl)

    errLast = 1e10

    for i in range(20):

        # -----------------------------
        # Compute Jacobian and residual
        # -----------------------------
        if useNumeric:
            Jac, residual = deriveErrNumeric(IRef, DRef, I, xi, Klvl)
        else:
            Jac, residual = deriveErrAnalytic(IRef, DRef, I, xi, Klvl)

        # -----------------------------
        # Remove invalid rows (NaN)
        # -----------------------------
        notValid = np.isnan(np.sum(Jac, axis=1))
        residual[notValid] = 0
        Jac[notValid, :] = 0

        # -----------------------------
        # Gauss–Newton step
        # -----------------------------
        H = Jac.T @ Jac
        b = Jac.T @ residual
        delta_xi = -np.linalg.solve(H, b)

        # -----------------------------
        # delta_xiate pose
        # xi = se3Log( se3Exp(delta_xi) * se3Exp(xi) )
        # -----------------------------
        lastXi = xi.copy()
        xi = se3Log(se3Exp(delta_xi.flatten()) @ se3Exp(xi.flatten())).reshape(6, 1)

        print("xi =", xi.ravel())

        # -----------------------------
        # Compute mean error
        # -----------------------------
        err = np.mean(residual * residual)
        print("error =", err)

        plt.pause(0.1)

        # -----------------------------
        # Early stopping
        # -----------------------------
        if err / errLast > 0.995:
            break

        errLast = err

