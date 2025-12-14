from calcErr import calcErr, se3Exp
import numpy as np
import numpy as np
from scipy.linalg import logm
import cv2

def se3Log(T):
    lg = logm(T)
    twist = np.array([
        lg[0, 3],  # tx
        lg[1, 3],  # ty
        lg[2, 3],  # tz
        lg[2, 1],  # wx
        lg[0, 2],  # wy
        lg[1, 0]   # wz
    ])
    return twist

def deriveErrNumeric(IRef, DRef, I, xi, K):
    # calculate numeric derivative (SLOW!!!)

    # compute residuals for xi
    residual_xi = calcErr(IRef,DRef,I,xi,K)

    H_I, W_I = I.shape[0], IRef.shape[1] # resolution of the reference image, heght and width
    num_of_pixels = H_I * W_I
    # initialize Jacobian
    Jac = np.zeros((num_of_pixels, 6))

    # compute Jacobian numerically
    eps = 1e-6
    for j in range(1, 7):
        epsVec = np.zeros((6, 1))
        epsVec[j-1] = eps
        # multiply epsilon from left onto the current estimate.
        xiPerm =  se3Log(se3Exp(epsVec) @ se3Exp(xi))
        # TODO compute respective column of the Jacobian
        # (hint: difference between residuals of xi and xiPerm)
        residual_perm = calcErr(IRef, DRef, I, xiPerm, K)
        Jac[:, j-1] = (residual_perm - residual_xi).flatten() / eps
    return Jac, residual_xi

def test(): # for testing
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from downscale import downscale

    # use cv2.imread to convert grayscale images as float
    # because map_coordinates requires float inputs
    IRef = cv2.imread("rgb/1305031102.175304.png", cv2.IMREAD_GRAYSCALE).astype(float)
    DRef = cv2.imread("depth/1305031102.160407.png", cv2.IMREAD_GRAYSCALE).astype(float)

    I = cv2.imread("rgb/1305031102.275326.png", cv2.IMREAD_GRAYSCALE).astype(float) # 2nd image

    # camera intrinsics
    K = np.array([[525.0, 0.0, 319.5],
                  [0.0, 525.0, 239.5],
                  [0.0, 0.0, 1.0]])

    # some arbitrary pose difference
    xi = np.array([0.05, 0.1, 0.2, 0.02, 0.02, 0.1]).reshape((6,1))

    # compute analytic and numeric Jacobian
    J, residual = deriveErrNumeric(IRef, DRef, I, xi, K)

    print("Numeric Jacobian shape: {}".format(J.shape))
    print("Numeric Jacobian:\n{}".format(J))
    print("Maximum element of Jacobian: {}".format(np.max(np.abs(J))))
    print("Norm of residuals: {}".format(np.linalg.norm(residual)))

if __name__ == "__main__":
    test()


