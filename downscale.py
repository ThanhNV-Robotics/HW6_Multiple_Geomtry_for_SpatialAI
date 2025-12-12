import numpy as np
import matplotlib.image as mpimg

from matplotlib import pyplot as plt

def downscale(I, D, K, level):
    """
    Recursively downscale intensity image, depth map, and camera intrinsics.

    Parameters
    ----------
    I : np.ndarray
        Intensity image (H x W)
    D : np.ndarray
        Depth map (H x W)
    K : np.ndarray
        Camera intrinsics (3 x 3)
    level : int
        Pyramid level (1 = coarsest / stop)

    Returns
    -------
    Id, Dd, Kd : downscaled I, D, K
    """

    # Base case: stop when level==1
    if level <= 1:
        #coarsest pyramid level
        return I, D, K

    # --- Downscale camera intrinsics ---
    # formular from slide 13
    Kd = K.copy()
    Kd[0, 0] *= 0.5   # fx/2
    Kd[1, 1] *= 0.5   # fy/2
    Kd[0, 2] = 0.5 * (Kd[0, 2]+0.5) - 0.5   # 0.5*(cx+0.5)-0.5
    Kd[1, 2] = 0.5 * (Kd[1, 2]+0.5) - 0.5   # 0.5*(cy+0.5)-0.5

    # --- Downscale intensity image (bilinear) ---
    H,W = I.shape[0], I.shape[1] # resolution of the input image
    # New resolution
    H2 = H // 2
    W2 = W // 2

    # Initialize downscaled intensity image
    Id = np.zeros((H2, W2), dtype=I.dtype)
    # Vectorize from equation (1)
    Id = 0.25 * (
        I[0:H:2, 0:W:2] + # I[0:H:2, 1:W:2] 
        I[1:H:2, 0:W:2] +
        I[0:H:2, 1:W:2] +
        I[1:H:2, 1:W:2]
    )

    # --- Downscale depth image
    # initialize downscaled depth image
    Dd = np.zeros((H2, W2), dtype=D.dtype)

    # Extract 2×2 blocks, this is O(x,y) set
    d00 = D[0:H:2, 0:W:2]
    d10 = D[1:H:2, 0:W:2]
    d01 = D[0:H:2, 1:W:2]
    d11 = D[1:H:2, 1:W:2]

    # Stack into (H2, W2, 4), then each pixel has 4 depth values
    blocks = np.stack([d00, d10, d01, d11], axis=-1)

    # Valid depth mask (non-zero) (H2, W2, 4) shape
    valid_mask = blocks != 0 # check which pixel has valid depth, returns boolean (0/1) array

    # Sum of valid depth values
    valid_sum = np.sum(blocks * valid_mask, axis=-1)

    # Count of valid values (H2, W2) shape
    valid_count = np.sum(valid_mask, axis=-1)

    # Avoid division by zero → depth stays 0
    nonzero = valid_count > 0

    # Take average of valid depth values as in equation (2)
    Dd[nonzero] = valid_sum[nonzero] / valid_count[nonzero]


    # --- Recursive call ---
    return downscale(Id, Dd, Kd, level - 1)

def test(): # for testing
    rgb_image_path = "rgb/1305031102.275326.png"
    depth_image_path = "depth/1305031102.160407.png"
    # load the image
    I = mpimg.imread(rgb_image_path)
    D = mpimg.imread(depth_image_path)

    # camera matrix
    K = np.array([[517.3, 0, 318.6],
                   [0, 516.5, 255.3],
                   [0, 0, 1]])
    
    downscale_level = [2,3,4] # test with different resolutions
    I_d = []
    D_d = []
    for level in downscale_level:
        img_d, depth_d, K_d = downscale(I, D, K, level=level)
        I_d.append(img_d)
        D_d.append(depth_d)

    # display the original image
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # display the downscaled image
    plt.subplot(2, 2, 2)
    plt.imshow(I_d[0], cmap='gray')
    plt.title('Downscaled Image: Level {}'.format(downscale_level[0]))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(I_d[1], cmap='gray')
    plt.title('Downscaled Image: Level {}'.format(downscale_level[1]))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(I_d[2], cmap='gray')
    plt.title('Downscaled Image: Level {}'.format(downscale_level[2]))
    plt.axis('off')

    # plot depth images
    plt.figure()    
    plt.subplot(2, 2, 1)
    plt.imshow(D, cmap='gray')
    plt.title('Original Depth')
    print("Original Depth shape: {}".format(D.shape))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(D_d[0], cmap='gray')
    plt.title('Downscaled Depth: Level {}'.format(downscale_level[0]))
    print("Downscaled Depth Level {} shape: {}".format(downscale_level[0], D_d[0].shape))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(D_d[1], cmap='gray')
    plt.title('Downscaled Depth: Level {}'.format(downscale_level[1]))
    print("Downscaled Depth Level {} shape: {}".format(downscale_level[1], D_d[1].shape))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(D_d[2], cmap='gray')
    plt.title('Downscaled Depth: Level {}'.format(downscale_level[2]))
    print("Downscaled Depth Level {} shape: {}".format(downscale_level[2], D_d[2].shape))
    plt.axis('off')

    # show all plots
    plt.show()

if __name__ == "__main__":
    test()

