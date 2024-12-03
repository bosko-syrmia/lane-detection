"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import cv2
from docopt import docopt

from camera_calibration import CalibrateCamera
from thresholding import Threshold
from perspective_transform import PerspectiveTransformation
from lane_detection import LaneDetection


class DetectLanes:

    def __init__(self):
        self.calibrate = CalibrateCamera(9, 6)
        self.threshold = Threshold()
        self.perspective_transform = PerspectiveTransformation()
        self.detect_lanes = LaneDetection()

    def forward(self, img):
        src_img = np.copy(img)
        img = self.calibrate(img)
        # img = self.threshold(img)
        # img = self.perspective_transform(img)
        # img = self.detect_lanes(img)
        # img = self.perspective_transform.backward(img)
        return img
        return cv2.addWeighted(src_img, 1, img, 0.6, 0)
    
    def process_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        if img.shape != (1280, 720):
            img = cv2.resize(img, (1280, 720))
        out_img = self.forward(img)
        res_img = cv2.resize(out_img, (960, 540))
        cv2.imwrite(output_path, res_img)
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, 0x14, fps, (1280, 720), True)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if size != (1280, 720):
                frame = cv2.resize(frame, (1280, 720))

            frame = self.forward(frame)
            out.write(frame)

            if (cv2.waitKey(30) & 0xff) == 27: break

        cap.release()
        out.release()
    

def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']

    detectLanes = DetectLanes()
    if args['--video']:
        detectLanes.process_video(input, output)
    else:
        detectLanes.process_image(input, output)


if __name__ == "__main__":
    main()



