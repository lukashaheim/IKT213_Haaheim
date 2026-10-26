import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

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

def SIFT_detection(reference_image, image_to_align, max_features, good_match_precent):
    img1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < good_match_precent*n.distance:
            good_matches.append(m)

    if len(good_matches) > max_features:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        image_to_align = cv2.polylines(image_to_align, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, good_matches, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()

def main():
    img1 = cv2.imread("./reference_img.png")
    img2 = cv2.imread("align_this.jpg", cv2.IMREAD_COLOR)

    # harris_corner_detection(img1)
    SIFT_detection(img1, img2, 10, 0.7)

if __name__ == "__main__":
    main()