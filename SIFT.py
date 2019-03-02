import cv2
import numpy as np

path = ''
img = cv2.imread(path + 'IMG_2009.jpg')
img=cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

print(img[10,10])
n_kp=100
sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)
#surf = cv2.xfeatures2d.SURF_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

img=cv2.drawKeypoints(img,keypoints,None)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()