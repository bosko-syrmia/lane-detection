import cv2
import numpy as np

def binary_threshold(img):
    sobel_kernel = 5
    mag_thresh = (10, 255)
    sat_thresh = (170, 255)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    s_binary = sobel_binary
    combined_binary = s_binary.astype(np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = 0 
    sobel_abs = np.abs(sobelx**2 + sobely**2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))

    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1
    s_binary[(s_channel >= sat_thresh[0]) & (s_channel <= sat_thresh[1])] = 1

    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))

    offset = 100
    mask_polyg = np.array([[(0 + offset, img.shape[0]),
                            (img.shape[1] / 2.5, img.shape[0] / 1.65),
                            (img.shape[1] / 1.8, img.shape[0] / 1.65),
                            (img.shape[1], img.shape[0])]],
                          dtype=np.int32)

    mask_img = np.zeros_like(combined_binary)
    ignore_mask_color = 255

    cv2.fillPoly(mask_img, mask_polyg, ignore_mask_color)
    masked_edges = cv2.bitwise_and(combined_binary, mask_img)

    return masked_edges

class Threshold:
    def __init__(self):
        pass

    def __call__(self, img):
        return binary_threshold(img)