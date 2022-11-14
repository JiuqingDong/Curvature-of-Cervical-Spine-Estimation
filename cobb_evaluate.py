###########################################################################################
## This code is transfered from matlab version of the MICCAI challenge
## Oct 1 2019
###########################################################################################
import numpy as np
import cv2


def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True


def cobb_angle_calc(pts, image):  # 传入 点的坐标、图片
    pts = np.asarray(pts, np.float32)   # 24 x 2

    #for pt in pts:
    #     cv2.circle(image,
     #               (int(pt[0]), int(pt[1])),
     #               4, (0,256,256), 3, 1)
    h,w,c = image.shape  # height width channel
    num_pts = pts.shape[0]   # number of points, 68

    vec_m = pts[1::2,:]-pts[0::2,:]           # 12 x 2              向量

    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 12 x 12             内积 a * b
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 12 x 1 这是向量的平方和再开方，模长，np.newaxis的作用就是在这一位置增加一个一维
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17     # 模长的积 |a|*|b|
    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)    # cos_angle = a*b/|a|*|b|
    angles = np.arccos(cosine_angles)   # 12 x 12
    cobb_angle1 = angles[1,11]
    cobb_angle1 = cobb_angle1/np.pi*180
    '''
    cv2.line(image,
             (int(pts[1 * 2, 0]) * 2 - int(pts[1 * 2 + 1, 0]),
              int(pts[1 * 2, 1]) * 2 - int(pts[1 * 2 + 1, 1])),
             (int(pts[1 * 2 + 1, 0]) * 2 - int(pts[1 * 2, 0]),
              int(pts[1 * 2 + 1, 1]) * 2 - int(pts[1 * 2, 1])),
             color=(0, 255, 255), thickness=4, lineType=2)
    cv2.line(image,
             (int(pts[11 * 2, 0]) * 2 - int(pts[11 * 2 + 1, 0]),
              int(pts[11 * 2, 1]) * 2 - int(pts[11 * 2 + 1, 1])),
             (int(pts[11 * 2 + 1, 0]) * 2 - int(pts[11 * 2, 0]),
              int(pts[11 * 2 + 1, 1]) * 2 - int(pts[11 * 2, 1])),
             color=(0, 255, 255), thickness=4, lineType=2)
    cv2.putText(image, "Cobb angle  %.2f" % cobb_angle1, (10, int((int(pts[1 * 2, 1]) + int(pts[11 * 2, 1])) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 2)
    ''' 
    return [cobb_angle1], image
