import numpy as np
from scipy.ndimage import map_coordinates
from scipy.linalg import expm
import cv2

def se3Exp(twist):
    M = np.array([[0, -twist[5].item(), twist[4].item(), twist[0].item()],
                  [twist[5].item(), 0, -twist[3].item(), twist[1].item()],
                  [-twist[4].item(), twist[3].item(), 0, twist[2].item()],
                  [0, 0, 0, 0]])
    return expm(M)

def calcErr(IRef, DRef, I, xi, K):
    # calculate residuals

    # get shorthands (R, t)
    T = se3Exp(xi)
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    K_inv = np.linalg.inv(K)
    H_ref, W_ref = IRef.shape[0], IRef.shape[1] # resolution of the reference image, heght and width
    # these contain the x,y image coordinates of the respective
    # reference-pixel, transformed & projected into the new image.
    # set to -10 initially, as this will give NaN on interpolation later.
    xImg = np.full((H_ref, W_ref), -10.0)
    yImg = np.full((H_ref, W_ref), -10.0)

    # for all pixels
    for x in range(W_ref):
        for y in range(H_ref):
            # TODO warp reference points to target image
            d = DRef[y, x]
            if d == 0:
                continue; # skip invalid depth
            # backproject to 3D point (3,1)
            temp = K@(R@K_inv @ np.array([x, y, 1])*d + t)
            w = np.array([temp[0]/temp[2], temp[1]/temp[2]]) # [x/z, y/z]
            # TODO project warped points onto image
            xImg[y, x] = w[0]
            yImg[y, x] = w[1]

 # --- 4. Bilinear interpolation, replacement for interp2
    coords = np.array([yImg.ravel(), xImg.ravel()])

    I_warped = map_coordinates(I, coords, order=1, mode='nearest')
    I_warped = I_warped.reshape(H_ref, W_ref)

    # --- 5. Photometric residual
    err = IRef - I_warped

    return err.reshape(-1, 1)

def test(): # for testing
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from downscale import downscale

    # use cv2.imread to convert grayscale images as float
    # because map_coordinates requires float inputs
    IRef = cv2.imread("rgb/1305031102.175304.png", cv2.IMREAD_GRAYSCALE).astype(float)
    DRef = cv2.imread("depth/1305031102.160407.png", cv2.IMREAD_GRAYSCALE).astype(float)

    I = cv2.imread("rgb/1305031102.275326.png", cv2.IMREAD_GRAYSCALE).astype(float) # new image

    # camera intrinsics
    K = np.array([[525.0, 0.0, 319.5],
                  [0.0, 525.0, 239.5],
                  [0.0, 0.0, 1.0]])
    
    # camera pose 6D vector
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #
    err = calcErr(IRef, DRef, I, xi, K)

    #visualize error as image
    err_image = err.reshape(IRef.shape)
    plt.imshow(err_image, cmap='gray')
    plt.title('Photometric Error Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test()

    