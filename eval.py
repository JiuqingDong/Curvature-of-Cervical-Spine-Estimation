import torch
import numpy as np
from models import spinal_net
import decoder
import os
from dataset import BaseDataset
import time
import cobb_evaluate
import cobb_gt
import cv2
import matplotlib.pyplot as plt
import pandas as pd

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

                 }  # 最后输出的三个字典 hm：heatmap reg：中心点偏移 wh：中心点指向四个角的向量

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


    def eval(self, args, save):
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

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        img_list = []
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            img_list.append(img_id)
            # print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output,_,   dec_dict_cat, dec_dict_de_c4,  dec_dict_de_c3 = self.model(images)
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
            pts2 = self.decoder.ctdet_decode(hm, heat_tl, heat_bl, heat_tr, heat_br, wh, reg, reg_tl, reg_bl, reg_tr, reg_br)   # 17, 11  #
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            for pr_pt, gt_pt in zip(pr_landmarks[2:], gt_landmarks[2:]):
                landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            calculate_pre, image = cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image)
            calculate_gt, image = cobb_gt.cobb_angle_calc(gt_landmarks, image)
            pr_cobb_angles.append(calculate_pre)
            gt_cobb_angles.append(calculate_gt)

            # cv2.imwrite('./imgs/gt_cobb/' + img_id, image)

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = np.sum(out_abs, axis=1)
        term2 = np.sum(out_add, axis=1)

        SMAPE = np.mean(term1 / term2 * 100)

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))
        print("==="*10)
        print('SMAPE is {}'.format(SMAPE))
        print("==="*10)

        total_time = total_time[1:]
        #print('avg time is {}'.format(np.mean(total_time)))
        #print('FPS is {}'.format(1./np.mean(total_time)))

        line_x = [1, 50]
        line_y = [1, 50]
        Cobb_gt = gt_cobb_angles[:,0]
        Cobb_pr = pr_cobb_angles[:,0]

        cobb_corelation = np.corrcoef(Cobb_gt, Cobb_pr)
        print("correlation is ", cobb_corelation[0,1])
        plt.plot(line_x, line_y,c='gray')
        plt.scatter(Cobb_gt, Cobb_pr, c='r',marker='x')
        plt.title('CS_799(correlation:'+'{:.4f})'.format(cobb_corelation[0,1]))
        plt.xlabel('Ground truth angle($^\circ$)', color='#1C2833', fontsize=12)
        plt.ylabel('Prediction angle($^\circ$)', color='#1C2833', fontsize=12)
        plt.savefig('./imgs/cobb_accuracy.jpg')
        plt.clf()
        plt.close()

        bias_cobb = Cobb_pr - Cobb_gt

        data = []
        data.append(img_list)
        data.append(Cobb_gt)
        data.append(Cobb_pr)
        data.append(bias_cobb)
        data = np.array(data)
        data_reshape = data.T
        data_csv = pd.DataFrame(data_reshape)
        data_csv.to_csv('./imgs/bias.csv')

        bias_tx1 = [0, 50]
        bias_tx2 = [0, 50]

        plt.plot(bias_tx1, (np.mean(np.absolute(bias_cobb)),np.mean(np.absolute(bias_cobb))), c='blue', linestyle='dashed')
        y_b = np.mean(np.absolute(bias_cobb))
        print('mean is ', y_b)
        plt.text(6, y_b, 'Mean: %.2f$^\circ$'%y_b, ha='center', va='bottom', fontsize=12)
        #plt.plot(bias_tx2, ((np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-5]]+np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-6]])/2,
        #                    (np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-5]]+np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-6]])/2), c='green', linestyle='dashed')
        abs_ = np.absolute(bias_cobb)
        std_ = np.std(abs_, ddof=1)
        line_std = y_b+std_*1.96
        print("std is ", std_)
        plt.plot(bias_tx2, (line_std, line_std), c='green', linestyle='dashed')

        # y = (np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-5]]+np.absolute(bias_cobb)[np.argsort(np.absolute(bias_cobb))[-6]])/2
        print('95% is ', line_std)
        plt.text(14, line_std, '95%'+' Confidence interval: %.2f$^\circ$'%line_std, ha='center', va='bottom', fontsize=12)
        plt.scatter(Cobb_pr, np.absolute(bias_cobb), c='r', marker='x')
        plt.title('CS_799')
        plt.xlim((0,50))
        plt.ylim((0,10))
        plt.xlabel('Mean measurement ($^\circ$)', color='#1C2833', fontsize=12)
        plt.ylabel('Absolute measurement differences($^\circ$)', color='#1C2833', fontsize=12)
        plt.savefig('./imgs/cobb_absolute.jpg')
        plt.clf()
        plt.close()
