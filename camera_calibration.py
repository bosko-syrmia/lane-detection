import numpy as np
import cv2
import glob

class CalibrateCamera():
    def __init__(self, nx, ny):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane

        images = glob.glob('camera_cal/*.jpg')
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def __call__(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)