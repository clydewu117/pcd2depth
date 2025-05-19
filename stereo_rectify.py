import cv2
import numpy as np


left_img, right_img = cv2.imread(left_src), cv2.imread(right_src)
R = np.matmul(np.linalg.inv(calibL.R), calibR.R)
T = np.matmul(np.linalg.inv(calibL.R), (calibR.T - calibL.T))
distCoeff = np.zeros(4)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=calibL.K[:3, :3],
    distCoeffs1=distCoeff,
    cameraMatrix2=calibR.K[:3, :3],
    distCoeffs2=distCoeff,
    imageSize=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
    R=R,
    T=T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(
    cameraMatrix=calibL.K[:3, :3],
    distCoeffs=distCoeff,
    R=R1,
    newCameraMatrix=P1,
    size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
    m1type=cv2.CV_32FC1)

map2x, map2y = cv2.initUndistortRectifyMap(
    cameraMatrix=calibR.K[:3, :3],
    distCoeffs=distCoeff,
    R=R2,
    newCameraMatrix=P2,
    size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
    m1type=cv2.CV_32FC1)

left_img_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
right_img_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
cv2.imwrite(left_dst, left_img_rect)
cv2.imwrite(right_dst, right_img_rect)