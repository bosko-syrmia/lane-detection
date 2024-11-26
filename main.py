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


class DetectLanes:

    def __init__(self):
        self.calibrate = CalibrateCamera(9, 6)
        self.threshold = Threshold()

    def forward(self, img):
        src_img = np.copy(img)
        img = self.calibrate(img)
        img = self.threshold(img)

        return img
    
    def process_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        out_img = self.forward(img)
        cv2.imwrite(output_path, out_img)
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, 0x14, fps, size, True)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
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



