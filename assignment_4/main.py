import cv2

img1 =  cv2.imread("./reference_img.png", cv2.IMREAD_COLOR)
img2 = cv2.imread("align_this.jpg", cv2.IMREAD_COLOR)

h, w, c = img1.shape
print('Image 1')
print('width: ', w)
print('height: ', h)
print('channel: ', c, '\n')

h, w, c = img2.shape
print('Image 2')
print('width: ', w)
print('height: ', h)
print('channel: ', c)

cv2.imshow("img1", img1)
cv2.waitKey(0)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



