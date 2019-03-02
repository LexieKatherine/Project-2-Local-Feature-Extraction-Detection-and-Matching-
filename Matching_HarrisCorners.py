import cv2
import numpy as np
import sift as sift

filename = 'IMG_2004.jpg'
matchfile= 'IMG_2009.jpg'
img = cv2.imread(filename)
match = cv2.imread(matchfile)
img=cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
match=cv2.resize(match,(0,0),fx=0.1,fy=0.1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
matchgray=cv2.cvtColor(match,cv2.COLOR_BGR2GRAY)
bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)

sift = cv2.xfeatures2d.SIFT_create()
corners = cv2.cornerHarris(gray,2,0,0.04)
kpsCorners = np.argwhere(corners>0.1*corners.max())
kpsCorners = [cv2.KeyPoint(pt[1],pt[0],3) for pt in kpsCorners]
grayWithCorners = cv2.drawKeypoints(gray,kpsCorners,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
kpsCorners,dscCorners = sift.compute(gray,kpsCorners)
kp = sift.detect(gray,None)
grayWithSift =cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp,dsc =sift.compute(gray,kp)

corners = cv2.cornerHarris(matchgray,2,0,0.04)
kpsCorners2 = np.argwhere(corners>0.1*corners.max())
kpsCorners2 = [cv2.KeyPoint(pt[1],pt[0],3) for pt in kpsCorners2]
#grayWithCorners = cv2.drawKeypoints(gray,kpsCorners,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
kpsCorners2,dscCorners2 = sift.compute(matchgray,kpsCorners2)

matchesCorners = bfmatcher.match(dscCorners,dscCorners2)
matchesCorners = sorted(matchesCorners,key=lambda  x:x.distance)
img3 = cv2.drawMatches(img,kpsCorners,match,kpsCorners2,matchesCorners[:10],None,flags=2)

cv2.imshow("match",img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()