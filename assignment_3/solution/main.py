import cv2
import numpy as np
import sys

def sobel_edge_detection(image):

    cv2.imshow('original', image)
    cv2.waitKey(0)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_grey, (3, 3), 0)

    # Sobel edge detection
    sobelxy = cv2.Sobel(src=image_blur, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobelxy = cv2.convertScaleAbs(sobelxy)

    cv2.imshow('sobel edge detection', sobelxy)
    cv2.waitKey(0)

    cv2.imwrite('./images/sobel_edge_detection.png', sobelxy)

def canny_edge_detection(image, threshold_1, threshold_2):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_grey, (3, 3), 0)

    edges = cv2.Canny(image=image_blur, threshold1=threshold_1, threshold2=threshold_2)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    cv2.imwrite('./images/canny_edge_detection.png', edges)

def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    treshold = 0.9
    loc = np.where( res >= treshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    cv2.imwrite('./images/template_match.png', image)

def resize(image, scale_factor:int, up_or_down:str):

    rows, cols, _channels = map(int, image.shape)

    if up_or_down == "up":
        image = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
        print('Zoom In: Image x 2')
    elif up_or_down == 'down':
        image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
        print('Zoom Out: Image x 2')
    cv2.imshow('Pyramids Demo', image)
    cv2.waitKey(0)
    cv2.imwrite('./images/pyramids_demo_'+up_or_down+'.png', image)


def main():
    image = cv2.imread('./images/lambo.png')
    sobel_edge_detection(image)

    canny_edge_detection(image, threshold_1=50, threshold_2=50)

    image = cv2.imread('./images/shapes-1.png')
    template = cv2.imread('./images/shapes_template.jpg', 0)
    template_match(image, template)

    image = cv2.imread('./images/golden-retriever-tongue-out.jpg')
    up_or_down = "down"
    resize(image, scale_factor=2, up_or_down=up_or_down)



    cv2.destroyAllWindows()


main()