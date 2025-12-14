import numpy as np
from scipy.ndimage import map_coordinates
from calcErr import se3Exp


def deriveErrAnalytic(IRef, DRef, I, xi, K):
    """
    Analytic Jacobian of photometric error
    """

    # --- SE(3)
    T = se3Exp(xi.flatten())
    R = T[:3, :3]
    t = T[:3, 3]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)

    H, W = IRef.shape # resolution of the reference image
    N = H * W # number of pixels

    # --- warped pixel coordinates
    xImg = np.full((H, W), -10.0)
    yImg = np.full((H, W), -10.0)

    # --- warped 3D points
    xp = np.full((H, W), np.nan)
    yp = np.full((H, W), np.nan)
    zp = np.full((H, W), np.nan)

    # =========================
    # Warp pixels
    # =========================
    for x in range(W):
        for y in range(H):

            d = DRef[y, x] # depth value on depth image
            if d == 0:
                continue

            # backproject
            p = d * (Kinv @ np.array([x, y, 1.0]))

            # transform
            Pw = R @ p + t
            Xp, Yp, Zp = Pw

            if Zp <= 0:
                continue

            # project
            u = fx * Xp / Zp + cx
            v = fy * Yp / Zp + cy

            xImg[y, x] = u
            yImg[y, x] = v

            xp[y, x] = Xp
            yp[y, x] = Yp
            zp[y, x] = Zp

    # =========================
    # Image gradients (central differences)
    # =========================
    dxI = np.zeros_like(I)
    dyI = np.zeros_like(I)

    dxI[:, 1:-1] = 0.5 * (I[:, 2:] - I[:, :-2]) # exclude 1st and last column
    dyI[1:-1, :] = 0.5 * (I[2:, :] - I[:-2, :]) # exclude 1st and last row

    # =========================
    # Interpolate gradients
    # =========================
    coords = np.vstack([yImg.ravel(), xImg.ravel()])

    dxInterp = fx * map_coordinates(dxI, coords, order=1, mode="nearest")
    dyInterp = fy * map_coordinates(dyI, coords, order=1, mode="nearest")

    # =========================
    # Flatten warped 3D points
    # =========================
    xp = xp.ravel()
    yp = yp.ravel()
    zp = zp.ravel()

    # =========================
    # Analytic Jacobian (Kerl Eq. 4.14)
    # =========================
    Jac = np.zeros((N, 6))

    inv_z = 1.0 / zp
    inv_z2 = inv_z ** 2

    Jac[:, 0] = dxInterp * inv_z
    Jac[:, 1] = dyInterp * inv_z
    Jac[:, 2] = -(dxInterp * xp + dyInterp * yp) * inv_z2

    Jac[:, 3] = -(dxInterp * xp * yp) * inv_z2 - dyInterp * (1 + (yp * inv_z) ** 2)
    Jac[:, 4] = dxInterp * (1 + (xp * inv_z) ** 2) + dyInterp * xp * yp * inv_z2
    Jac[:, 5] = -(dxInterp * yp - dyInterp * xp) * inv_z

    # Invert Jacobian sign 
    Jac = -Jac

    # =========================
    # Residual
    # =========================
    Iwarp = map_coordinates(I, coords, order=1, mode="nearest")
    residual = (IRef.ravel() - Iwarp).reshape(-1, 1)

    return Jac, residual