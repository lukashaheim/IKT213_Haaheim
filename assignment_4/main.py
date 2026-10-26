import cv2
import numpy as np

def harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    #dst = cv2.dilate(dst, None)

    # threshold for an optimal value
    reference_image[dst>0.01*dst.max()] = [0, 0, 255]

    cv2.imshow("dst", reference_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./results/harris.png", reference_image)

def main():
    img1 = cv2.imread("./reference_img.png")
    img2 = cv2.imread("align_this.jpg", cv2.IMREAD_COLOR)

    harris_corner_detection(img1)

if __name__ == "__main__":
    main()