import os
import sys
import cv2
import numpy as np
import time

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):

        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        f1 = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        f2 = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)

        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f1, 0, cx],
                [0, f2, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        print(time.time() - start)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        print(time.time() - start)

        xyz = xyz @ K_inv.T
        print(time.time() - start)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, t = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, t = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        print(time.time() - start)
        xyz = xyz @ R.T
        print("Passo 10: %.4f"%(time.time() - start))

        lonlat = xyz2lonlat(xyz)
        print(time.time() - start)

        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        print(time.time() - start)

        return persp

    @property
    def img(self):
        return self._img


e = Equirectangular("src/image.jpg")

h = e._height//5;
w = e._width//5;
r2 = 5

# for i in range( e._height ):
#     for j in range( e._width//r2):
#         e.img[i][j] = [0, 255, 0]
#     for j in range( e._width - e._width//r2, e._width):
#         e.img[i][j] = [0, 255, 0]
#
# img2 = cv2.resize(e.img, (w, h), interpolation = cv2.INTER_AREA);
# cv2.imshow('image', img2)
# cv2.waitKey(0)

start = time.time()

for i in range(10):
    print('Init:')
    start = time.time()
    img = e.GetPerspective( 90, i, 0, 720, 1024)
    print(time.time() - start)
    print("Fim")

print(time.time() - start)
cv2.imshow('image', img)
cv2.waitKey(0)


#cv2.imshow('image', img)
#cv2.waitKey(0)
