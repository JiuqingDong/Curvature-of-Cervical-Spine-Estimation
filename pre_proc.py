import cv2
import torch
from draw_gaussian import *
import transform
import math
NUM_BOX = 6

def processing_test(image, input_h, input_w):
    image = cv2.resize(image, (input_w, input_h))
    out_image = image.astype(np.float32) / 255.
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        cv2.putText(out_image, '{}'.format(i+1), (int(pts[i, 0]), int(pts[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0),1,1)
    for i,j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(out_image,
                 (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])),
                 color=colors[i], thickness=1, lineType=1)
    return out_image


def rearrange_pts(pts):
    # rearrange left right sequence
    boxes = []
    centers = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
        centers.append(np.mean(pts_4, axis=0))
    bboxes = np.asarray(boxes, np.float32)
    # rearrange top to bottom sequence
    centers = np.asarray(centers, np.float32)
    sort_tb = np.argsort(centers[:,1])
    new_bboxes = []
    for sort_i in sort_tb:
        new_bboxes.append(bboxes[4*sort_i, :])
        new_bboxes.append(bboxes[4*sort_i+1, :])
        new_bboxes.append(bboxes[4*sort_i+2, :])
        new_bboxes.append(bboxes[4*sort_i+3, :])
    new_bboxes = np.asarray(new_bboxes, np.float32)
    return new_bboxes


def generate_ground_truth(image,
                          pts_2,
                          image_h,
                          image_w,
                          img_id):
    hm = np.zeros((1, image_h, image_w), dtype=np.float32)
    heat_tl = np.zeros((1, image_h, image_w), dtype=np.float32)  # 想建个左上、右上、左下、右下 4个
    heat_bl = np.zeros((1, image_h, image_w), dtype=np.float32)
    heat_tr = np.zeros((1, image_h, image_w), dtype=np.float32)
    heat_br = np.zeros((1, image_h, image_w), dtype=np.float32)

    wh = np.zeros((NUM_BOX, 2*4), dtype=np.float32)
    reg = np.zeros((NUM_BOX, 2), dtype=np.float32)
    ind = np.zeros((NUM_BOX), dtype=np.int64)
    reg_mask = np.zeros((NUM_BOX), dtype=np.uint8)
    # print(pts_2.shape) # pts_2=[68,2]
    ind_tl = np.zeros((NUM_BOX), dtype=np.int64)
    reg_tl = np.zeros((NUM_BOX, 2), dtype=np.float32)
    reg_mask_tl = np.zeros((NUM_BOX), dtype=np.uint8)

    ind_bl = np.zeros((NUM_BOX), dtype=np.int64)
    reg_bl = np.zeros((NUM_BOX, 2), dtype=np.float32)
    reg_mask_bl = np.zeros((NUM_BOX), dtype=np.uint8)

    ind_tr = np.zeros((NUM_BOX), dtype=np.int64)
    reg_tr = np.zeros((NUM_BOX, 2), dtype=np.float32)
    reg_mask_tr = np.zeros((NUM_BOX), dtype=np.uint8)

    ind_br = np.zeros((NUM_BOX), dtype=np.int64)
    reg_br = np.zeros((NUM_BOX, 2), dtype=np.float32)
    reg_mask_br = np.zeros((NUM_BOX), dtype=np.uint8)

    if pts_2[:,0].max()>image_w:
        print('w is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())

    if pts_2.shape[0] != (NUM_BOX*4):
        print('ATTENTION!! image {} pts does not equal to 20!!! '.format(img_id))

    for k in range(NUM_BOX):
        pts = pts_2[4*k:4*k+4,:]
        pt_tl = pts_2[4*k,:]  # tl.shape=[17,2]
        pt_bl = pts_2[4*k+1,:]
        pt_tr = pts_2[4*k+2,:]
        pt_br = pts_2[4*k+3,:]
        # print(pts.shape)  # (4,2)
        bbox_h = np.mean([np.sqrt(np.sum((pts[0,:]-pts[2,:])**2)),
                          np.sqrt(np.sum((pts[1,:]-pts[3,:])**2))])
        bbox_w = np.mean([np.sqrt(np.sum((pts[0,:]-pts[1,:])**2)),
                          np.sqrt(np.sum((pts[2,:]-pts[3,:])**2))])
        cen_x, cen_y = np.mean(pts, axis=0)  # 各列求均值
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)

        pt_tl = np.asarray(pt_tl, dtype=np.float32)
        pt_tl_int = pt_tl.astype(np.int32)

        pt_bl = np.asarray(pt_bl, dtype=np.float32)
        pt_bl_int = pt_bl.astype(np.int32)

        pt_tr = np.asarray(pt_tr, dtype=np.float32)
        pt_tr_int = pt_tr.astype(np.int32)

        pt_br = np.asarray(pt_br, dtype=np.float32)
        pt_br_int = pt_br.astype(np.int32)

        # print(ct_int.shape)  # (2,)
        radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[0, :, :], ct_int, radius=radius)
        draw_umich_gaussian(heat_tl[0, :, :], pt_tl_int, radius=radius)
        draw_umich_gaussian(heat_bl[0, :, :], pt_bl_int, radius=radius)
        draw_umich_gaussian(heat_tr[0, :, :], pt_tr_int, radius=radius)
        draw_umich_gaussian(heat_br[0, :, :], pt_br_int, radius=radius)

        ind[k] = ct_int[1] * image_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1

        ind_tl[k] = pt_tl_int[1] * image_w + pt_tl_int[0]
        reg_tl[k] = pt_tl - pt_tl_int
        reg_mask_tl[k] = 1

        ind_bl[k] = pt_bl_int[1] * image_w + pt_bl_int[0]
        reg_bl[k] = pt_bl - pt_bl_int
        reg_mask_bl[k] = 1

        ind_tr[k] = pt_tr_int[1] * image_w + pt_tr_int[0]
        reg_tr[k] = pt_tr - pt_tr_int
        reg_mask_tr[k] = 1

        ind_br[k] = pt_br_int[1] * image_w + pt_br_int[0]
        reg_br[k] = pt_br - pt_br_int
        reg_mask_br[k] = 1

        for i in range(4):
            wh[k,2*i:2*i+2] = ct-pts[i,:]

    # corner = []
    # pt_tl = pts_2[0::4]  # tl.shape=[17,2]
    # pt_bl = pts_2[1::4]
    # pt_tr = pts_2[2::4]
    # pt_br = pts_2[3::4]
    # corner = np.asarray([], dtype=np.float32)
    # print(corner.size)  # 我期望的是[4,17,2]


    ret = {'input': image,
           'hm': hm,
           'ind': ind,
           'reg': reg,
           'wh': wh,
           'reg_mask': reg_mask,

           'heat_tl': heat_tl,
           'reg_tl': reg_tl,
           'reg_mask_tl': reg_mask_tl,
           'ind_tl': ind_tl,

           'heat_bl': heat_bl,
           'reg_bl': reg_bl,
           'reg_mask_bl': reg_mask_bl,
           'ind_bl': ind_bl,

           'heat_tr': heat_tr,
           'reg_tr': reg_tr,
           'reg_mask_tr': reg_mask_tr,
           'ind_tr': ind_tr,

           'heat_br': heat_br,
           'reg_br': reg_br,
           'reg_mask_br': reg_mask_br,
           'ind_br': ind_br,
           }

    return ret

# def filter_pts(pts, w, h):
#     pts_new = []
#     for pt in pts:
#         if any(pt) < 0 or pt[0] > w - 1 or pt[1] > h - 1:
#             continue
#         else:
#             pts_new.append(pt)
#     return np.asarray(pts_new, np.float32)


def processing_train(image, pts, image_h, image_w, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),
                                            transform.PhotometricDistort(),
                                            transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            transform.Equalize(),
                                            #transform.Color(),
                                            #transform.Posterize(),
                                            #transform.Solarize(),
                                            #transform.Sharpness(),
                                            transform.RandomMirror_w(),
                                            transform.Resize(h=image_h, w=image_w)]),
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(h=image_h, w=image_w)])}
    if aug_label:
        out_image, pts = data_aug['train'](image.copy(), pts)
    else:
        out_image, pts = data_aug['val'](image.copy(), pts)

    out_image = np.clip(out_image, a_min=0., a_max=255.)  # 将数组元素限制在a_min,a_max之间
    out_image = np.transpose(out_image / 255. - 0.5, (2,0,1))
    # print(out_image.shape)=(3, 1024, 512)
    pts = rearrange_pts(pts)
    pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)

    return np.asarray(out_image, np.float32), pts2

