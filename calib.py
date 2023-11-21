import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*11,3), np.float32)
infty = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])
objp[:,0:2] = np.mgrid[0:11,0:7].T.reshape(-1,2)

def calibrate(img_folder, dsize = None, insintric = None, dist = None):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray = []
    images = glob.glob( img_folder + '/*.png')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if dsize is not None:
            gray = cv.resize(gray, dsize)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11,7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # plt.imshow(gray)
            # plt.scatter(corners2[:, :, 0], corners2[:, :, 1])
            # plt.pause(2)
            imgpoints.append(corners2)
    if insintric is None:
        return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    else:
        return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], distCoeffs=dist,cameraMatrix=insintric)

def norm_z(dst, tvec): 
    htvec = np.append(tvec, 1).reshape([4,-1])
    ratio = - dst[2] / (-dst[2] + htvec[2])
    return htvec * ratio + dst * (1 - ratio)

def project(point3ds, mtx):
    output = np.array([np.matmul(mtx, point.reshape([4, -1])) for point in point3ds]).reshape([-1, 3])
    output = np.divide(output, output[:, 2].reshape([-1,1]))
    return output

def deproject(point2ds, mtx, dist):
    pass
import matplotlib.pyplot as plt

_, mtx_1, dist_1, rvec_1, tvec_1 = calibrate("data/trash", (700, 490))
mtx_1, roi = cv.getOptimalNewCameraMatrix(mtx_1, -dist_1, (700,490), 1, (700,490))
# print("Difference: ", nmtx_1 - mtx_1)
_, mtx_2, dist_2, rvec_2, tvec_2 = calibrate("data/imgs/leftcamera", (700, 490), insintric=mtx_1, dist=dist_1)
print(dist_1)
rvec_1 = np.average(rvec_1, axis=0)
rotate_1, _ = cv.Rodrigues(rvec_1)
# rotate_1 = np.matmul(np.array(rotate_1), np.transpose([1, 1, 1])) * 10
# print(rotate_1)
# The importance of atribirary scalar is to scale :) 
mtx_1  = mtx_1
# print("Tvec :{}".format(tvec_1))
tvec_1 = np.average(tvec_1, axis=0)
print(tvec_1)
homo_l = np.matmul(mtx_1 , np.append(rotate_1, tvec_1, axis=1))
line_1 = - np.reshape(np.matmul(np.array(rotate_1), np.transpose([1, 1, 1])) * 10, (3,-1)) + tvec_1
# tvec_l = homo_l[:, 3].reshape([3,-1])
# line_1 = tvec_l - np.matmul(homo_l[:, 0:3], np.transpose((1, 1, 1))).reshape([3,-1])
# print(tvec_1)
# print(line_1)

def distort_mapping(dsize, dist, points):
    k1, k2, p1, p2, k3 = dist 
    # points[:, :2] -= dsize/2
    r = np.linalg.norm(points, axis = 1)
    r2 = r**2
    r4 = r**4
    r6 = r**6
    x = points[:, 0].copy()
    x2 = x**2
    y = points[:, 1].copy()
    y2 = y**2
    dr = k1 * r2 + k2 * r4 + k3 * r6
    dx = x * dr + 2 * p1 * x * y + p2 * (r2 + 2 * x2)
    dy = y * dr + 2 * P2 * x * y + p1 * (r2 + 2 * y2)
    return dx, dy


print("camera 1 matrix = \n{}".format(mtx_1))
print("camera 1 position = {}".format(tvec_1))
print("hompgraphy 1 matrix = \n{}".format(homo_l))

rvec_2 = np.average(rvec_2, axis=0)
rotate_2, _ = cv.Rodrigues(rvec_2)
# rotate_2 = np.matmul(np.array(rotate_2), np.transpose([1, 1, 1])) * 10
# print(rotate_2)
# pixel size is 1/18 mm
# there is a confusion since cameraMatrix is pixelated but not other metrics
mtx_2  = mtx_2 
tvec_2 = np.average(tvec_2, axis=0)
homo_r = np.matmul(mtx_2 , np.append(rotate_2, tvec_2, axis=1))
line_2 = - np.reshape(np.matmul(np.array(rotate_2), np.transpose([1, 1, 1])) * 10, (3,-1)) + tvec_2
# tvec_2 = homo_r[:, 3].reshape([3,-1])
# line_2 = tvec_2 - np.matmul(homo_r[:, 0:3], np.transpose((1, 1, 1))).reshape([3,-1])
# print(line_2)
print("camera 2 matrix = \n{}".format(mtx_2))
print("camera 2 position = {}".format(tvec_2))
print("hompgraphy 2 matrix = \n{}".format(homo_r))

hobjp = np.zeros([7 * 11,4], np.float32)
hobjp[:, 0:2] = np.mgrid[0:11,0:7].T.reshape(-1,2)
hobjp[:, 3] = 1
proj_l = project(hobjp, homo_l)
# convert from one camera to another
point_left = np.array([300, 300, 1], dtype=np.float32)
# print(homo_l)
# point_left -= homo_l[:, 3]
dehomo = np.delete(homo_l, 2, axis=1)
# dehomo=homo_l
pt_3_l = np.matmul(np.linalg.inv(dehomo), point_left.transpose())
pt_3_l = np.insert(pt_3_l, 2, 0) / pt_3_l[2]
pt_3_l = np.expand_dims(pt_3_l, axis=0)
print(pt_3_l)
point_3_l = point_left - homo_l[:, 3]
inv_mtx = np.linalg.inv(homo_l[:, 0:3])
point_3_l = np.matmul(inv_mtx, np.transpose(point_3_l)).reshape((3,-1))
point_3_l = np.append(point_3_l, [1]).reshape([4,-1])
# point_3_l = norm_z(point_3_l, tvec_1)
print("3d point = \n {}".format(point_3_l))
point_r = np.matmul(homo_r, np.array([5, 7, 0, 1]).reshape([4,-1]))
pt_r = project(pt_3_l, homo_r)
print("Right camera = \n{}".format(point_r))
ax1 = plt.subplot((131), projection="3d")
# ax1.plot([tvec_1[0], line_1[0]],
#          [tvec_1[1], line_1[1]],
#          [tvec_1[2], line_1[2]], label="left")

# ax1.plot([tvec_2[0], line_2[0]],
#          [tvec_2[1], line_2[1]],
#          [tvec_2[2], line_2[2]], label="right")

# ax1.scatter([tvec_1[0], tvec_2[0]],
#          [tvec_1[1], tvec_2[1]],
#          [tvec_1[2], tvec_2[2]])
ax1.scatter(pt_3_l[:, 0], pt_3_l[:, 1], pt_3_l[:, 2], marker='s', s=10)
ax1.plot_trisurf([0, 100, 100, 0], [0, 0, 100, 100], [0, 0, 0, 0])
ax1.legend()

from PIL import Image 
ax2 = plt.subplot((132))
img_l = Image.open("data/trash/left_test.png").resize((700, 490))
ax2.set_title("Left camera view")
ax2.imshow(img_l)
ax2.scatter(point_left[0], point_left[1], s=5)
ax2.scatter(proj_l[:, 0], proj_l[:, 1], s=5)

ax3 = plt.subplot((133))
img_r = Image.open("data/imgs/leftcamera/IM_L_1.png").resize((700, 490))
ax3.set_title("Right camera view")
ax3.imshow(img_r)
# ax3.scatter(point_r[0], point_r[1], s=5)
ax3.scatter(pt_r[:, 0], pt_r[:, 1], s= 5)
# ax3.scatter(proj_l[:, 0], proj_l[:, 1], s=5)
# ax3.scatter(projection_l[:, 0], projection_l[:, 1], s=5)


plt.show()
