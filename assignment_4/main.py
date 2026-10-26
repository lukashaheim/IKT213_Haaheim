import os

import cv2
import numpy as np

def harris_corner_detection(reference_image):
    copy = reference_image.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    # threshold for an optimal value
    copy[dst>0.01*dst.max()] = [0, 0, 255]

    cv2.imshow("dst", copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if os.path.isdir(os.getcwd() + "/results"):
        cv2.imwrite("./results/harris.png", copy)
    else:
        print("Results folder not found")

def main():
    img1 = cv2.imread("./reference_img.png")
    img2 = cv2.imread("align_this.jpg", cv2.IMREAD_COLOR)

    harris_corner_detection(img1)

if __name__ == "__main__":
    main()