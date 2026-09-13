import cv2
import numpy as np

def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )
    return padded_image

def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    return image[y_0:y_1, x_0:x_1]

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    return emptyPictureArray

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = (int(image[y, x, c]) + hue) % 256
    return emptyPictureArray

def smoothing(image):
    blurred_image = cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)
    return blurred_image

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180")
    return rotated_image

if __name__ == '__main__':
    image = cv2.imread('iris-1.png')

#asssignemnt 1
    if image is None:
        raise FileNotFoundError("image not found")

    padded = padding(image, border_width=100)

    cv2.imwrite('padding.png', padded)

#assignment 2
    height, width = image.shape[:2]

    x_0 = 200
    y_0 = 200
    x_1 = width - 130
    y_1 = height - 130

    cropped = crop(image, x_0, x_1, y_0, y_1)

    cv2.imwrite('cropped.png', cropped)
    print(f"Original: {width}x{height}")
    print(f"Cropped: {cropped.shape[1]}x{cropped.shape[0]}")

#assignment 3
    resized = resize(image, width=200, height=200)
    cv2.imwrite('resized.png', resized)
    print(f"Original: {image.shape[1]}x{image.shape[0]}")
    print(f"Resized: {resized.shape[1]}x{resized.shape[0]}")

#assignment 4
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copied_image = copy(image, emptyPictureArray)
    cv2.imwrite('copied_image.png', copied_image)

# assignment 5
    gray = grayscale(image)
    cv2.imwrite('grayscale.png', gray)

# assignment 6
    hsv_image = hsv(image)
    cv2.imwrite('hsv.png', hsv_image)

# assignment 7
    height, width, channels = image.shape
    emptyPictureArray_hue = np.zeros((height, width, channels), dtype=np.uint8)
    shifted = hue_shifted(image, emptyPictureArray_hue, hue=50)
    cv2.imwrite('hue_shifted.png', shifted)

# assignment 8
    smoothed = smoothing(image)
    cv2.imwrite('smoothing.png', smoothed)

# assignment 9
    rotated_90 = rotation(image, 90)
    cv2.imwrite('rotation_90.png', rotated_90)

    rotated_180 = rotation(image, 180)
    cv2.imwrite('rotation_180.png', rotated_180)

    print("Finnished")