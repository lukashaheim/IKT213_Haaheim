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

def SIFT_detection(image_to_align, reference_image, max_features, good_match_precent):
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
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    if len(good_matches) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H_ref_to_photo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        H_photo_to_ref, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(pts, H_ref_to_photo)

        photo_with_frame = cv2.polylines(image_to_align.copy(), [np.int32(corners)],
                                         True, (0, 0, 255), 3, cv2.LINE_AA)

        aligned = cv2.warpPerspective(image_to_align, H_photo_to_ref, (w, h))

        if os.path.isdir("results"):
            cv2.imwrite("./results/aligned.png", aligned)
        else:
            print("Results folder not found")
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), max_features))
        matchesMask = None
        photo_with_frame = image_to_align

    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(reference_image, kp1, photo_with_frame, kp2, good_matches, None, **draw_params)

    if os.path.isdir("results"):
        cv2.imwrite("./results/matches.png", img3)
    else:
        print("Results folder not found")


def main():
    align_this = cv2.imread("align_this.jpg")
    reference_img = cv2.imread("./reference_img.png")

    harris_corner_detection(reference_img)
    SIFT_detection(align_this, reference_img, 10, 0.7)

if __name__ == "__main__":
    main()