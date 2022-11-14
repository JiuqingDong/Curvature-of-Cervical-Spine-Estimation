import torch
import numpy as np
from models import spinal_net
import matplotlib.pyplot as plt
import cv2
import decoder
import os
from dataset import BaseDataset
import draw_points

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,
                 'heat_tl': args.num_classes,
                 'heat_bl': args.num_classes,
                 'heat_tr': args.num_classes,
                 'heat_br': args.num_classes,
                 'reg_tl': 2 * args.num_classes,
                 'reg_bl': 2 * args.num_classes,
                 'reg_tr': 2 * args.num_classes,
                 'reg_br': 2 * args.num_classes,
                 }

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)


    def test(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                _, output, dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3 = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']
                heat_tl = output['heat_tl']
                heat_bl = output['heat_bl']
                heat_tr = output['heat_tr']
                heat_br = output['heat_br']
                reg_tl = output['reg_tl']
                reg_bl = output['reg_bl']
                reg_tr = output['reg_tr']
                reg_br = output['reg_br']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, heat_tl, heat_bl, heat_tr, heat_br, wh, reg, reg_tl, reg_bl, reg_tr,
                                             reg_br)   # 17, 11 # heat_bl, heat_tr, heat_br,
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio

            print('totol pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
                                                                                          ori_image_regress,
                                                                                          ori_image_points)
            # hm2 = hm.cpu()
            # hm2 = np.squeeze(hm2)
            # # hm2 = np.asarray(hm2, np.float32) * 4
            # plt.matshow(hm2)
            # image2 = images.cpu()
            # # print(image2.size()) [1, 3, 1024, 512])
            # image2 = image2[0][1]
            # image2 = np.array(image2)
            # resize = cv2.resize(image2, (128, 256), interpolation=cv2.INTER_CUBIC)
            # # plt.figure(figsize=(1, 1))
            # # print(image2.size())
            # plt.imshow(resize, cmap='bone')
            # plt.imshow(hm2, cmap='jet', alpha=0.6)  # 蓝-青-黄-红
            # plt.show()

            cv2.imshow('ori_image_regress', ori_image_regress)
            cv2.imshow('ori_image_points', ori_image_points)

            # cv2.imwrite("./imgs/name%d.png", ori_image_regress)
            # cv2.imwrite("./imgs/name2%d.png", ori_image_points)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()
