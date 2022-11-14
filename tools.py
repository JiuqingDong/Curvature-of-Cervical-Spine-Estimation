from scipy.io import loadmat
import numpy as np
import cv2 as cv
import os


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]  # 四个点
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


def load_gt_pts(annopath):
    mat = loadmat(annopath)
    # print(mat)
    pts = mat['p2']   # num x 2 (x,y)
    pts = rearrange_pts(pts)
    # print(pts)
    return pts


def draw_gt_pts(points):
    img = cv.imread("./dataPath/data/val/sunhl-1th-01-Mar-2017-310 a ap.jpg")
    print(img.shape)
    for point in points:
        # print(point)
        cv.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
    img = cv.resize(img, (512, 1024))
    cv.imshow("gt", img)
    cv.waitKey(0)


def draw_gt_pts2(mat_path, jpg_path):
    img = cv.imread(jpg_path)
    points = load_gt_pts(mat_path)
    print(img.shape)
    i = 0
    for point in points:
        i += 1
        cv.circle(img, (point[0], point[1]), 1, (0,0,255), 2)
    img = cv.resize(img, (512, 1024))
    print("all points are:", end='')
    print(i)
    cv.imshow("gt", img)
    cv.waitKey(0)


def load_draw_gt_point(dir):
    files = os.listdir(dir)
    for file in files:
        if file.split('.')[1] == 'csv':
            continue
        mat_path = "./dataPath/labels/train/" + file
        name = file.split('.m')[0]
        jpg_path = "./dataPath/data/train/" + name
        print(mat_path, jpg_path)
        draw_gt_pts2(mat_path, jpg_path)


if __name__ == '__main__':
    annopath = './dataPath/labels/val/sunhl-1th-01-Mar-2017-310 a ap.jpg.mat'
    # pts = load_gt_pts(annopath)
    # draw_gt_pts(pts)
    dir = './dataPath/labels/train'
    load_draw_gt_point(dir)
