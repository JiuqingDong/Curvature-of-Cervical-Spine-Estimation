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
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            # print('processing {}/{} image ...'.format(cnt, len(data_loader)))

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
                                             reg_br)   # 17, 11  #
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
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))
            calculate_pre, image = cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image)
            calculate_gt, image = cobb_gt.cobb_angle_calc(gt_landmarks, image)
            pr_cobb_angles.append(calculate_pre)
            gt_cobb_angles.append(calculate_gt)

            cv2.imwrite('./imgs/landmarks/eval_angle'+img_id, image)

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
        # print(gt_cobb_angles[:,0], pr_cobb_angles[:,0])

        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = np.sum(out_abs, axis=1)
        term2 = np.sum(out_add, axis=1)
        SMAPE = np.mean(term1 / term2 * 100)

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))
        print("=="*30)
        print('SMAPE is {}'.format(SMAPE))
        print("=="*30)
        # total_time = total_time[1:]
        # print('avg time is {}'.format(np.mean(total_time)))
        # print('FPS is {}'.format(1./np.mean(total_time)))


    def SMAPE_single_angle(self, gt_cobb_angles, pr_cobb_angles):
        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = out_abs
        term2 = out_add

        term2[term2==0] += 1e-5

        SMAPE = np.mean(term1 / term2 * 100)
        return SMAPE

    def eval_three_angles(self, args, save):
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
        gt_cobb_angles = []
        img_list = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]

            img_list.append(img_id)
            images = images.to('cuda')
            # print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                _, output,dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3 = self.model(images)
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
                                             reg_br)   # 17, 11 #  heat_bl, heat_tr, heat_br,
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
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            calculate_pre, image = cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image)
            calculate_gt, image = cobb_gt.cobb_angle_calc(gt_landmarks, image)
            pr_cobb_angles.append(calculate_pre)
            gt_cobb_angles.append(calculate_gt)

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

        print('SMAPE1 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,0], pr_cobb_angles[:,0])))
        print('SMAPE2 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,1], pr_cobb_angles[:,1])))
        print('SMAPE3 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,2], pr_cobb_angles[:,2])))
        #print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))

        #total_time = total_time[1:]
        #print('avg time is {}'.format(np.mean(total_time)))
        #print('FPS is {}'.format(1./np.mean(total_time)))

        line_x = [1, 80]
        line_y = [1, 80]
        cobb1_gt = gt_cobb_angles[:,0]
        cobb1_pr = pr_cobb_angles[:,0]
        cobb2_gt = gt_cobb_angles[:,1]
        cobb2_pr = pr_cobb_angles[:,1]
        cobb3_gt = gt_cobb_angles[:,2]
        cobb3_pr = pr_cobb_angles[:,2]

        cobb1_corelation = np.corrcoef(cobb1_gt, cobb1_pr)
        plt.plot(line_x, line_y,c='gray')
        plt.scatter(cobb1_gt, cobb1_pr, c='r',marker='x', )
        plt.title('PT(corelation:'+'{:.4f})'.format(cobb1_corelation[0,1]))
        plt.xlabel('Ground Truth', color='#1C2833')
        plt.ylabel('Prediction', color='#1C2833')
        plt.savefig('./imgs/cobb1_accuracy.jpg')
        plt.clf()
        plt.close()

        cobb2_corelation = np.corrcoef(cobb2_gt, cobb2_pr)
        plt.plot(line_x, line_y,c='gray')
        plt.scatter(cobb2_gt, cobb2_pr, c='r',marker='x', )
        plt.title('MT(corelation:'+'{:.4f})'.format(cobb2_corelation[0,1]))
        plt.xlabel('Ground Truth', color='#1C2833')
        plt.ylabel('Prediction', color='#1C2833')
        plt.savefig('./imgs/cobb2_accuracy.jpg')
        plt.clf()
        plt.close()

        cobb3_corelation = np.corrcoef(cobb3_gt, cobb3_pr)
        plt.plot(line_x, line_y,c='gray')
        plt.scatter(cobb3_gt, cobb3_pr, c='r',marker='x', )
        plt.title('TL(corelation:'+'{:.4f})'.format(cobb3_corelation[0,1]))
        plt.xlabel('Ground Truth', color='#1C2833')
        plt.ylabel('Prediction', color='#1C2833')
        plt.savefig('./imgs/cobb3_accuracy.jpg')
        plt.clf()
        plt.close()
        bias_cobb1 = cobb1_pr - cobb1_gt
        bias_cobb2 = cobb2_pr - cobb2_gt
        bias_cobb3 = cobb3_pr - cobb3_gt

        data = []
        data.append(img_list)
        data.append(cobb1_gt)
        data.append(cobb1_pr)
        data.append(bias_cobb1)
        data.append(cobb2_gt)
        data.append(cobb2_pr)
        data.append(bias_cobb2)
        data.append(cobb3_gt)
        data.append(cobb3_pr)
        data.append(bias_cobb3)

        data = np.array(data)
        data_reshape = data.T
        data_csv = pd.DataFrame(data_reshape)
        data_csv.to_csv('./imgs/bias.csv')


        bias_x = [0,80]
        bias_y = [0,0]
        bias_tx1 = [0, 80]
        bias_ty1 = [5, 5]
        bias_bx1 = [0, 80]
        bias_by1 = [-5, -5]

        bias_tx2 = [0, 80]
        bias_ty2 = [10, 10]
        bias_bx2 = [0, 80]
        bias_by2 = [-10, -10]

        plt.plot(bias_x, bias_y, c='yellow', linestyle='dashed')
        plt.plot(bias_tx1, bias_ty1, c='blue', linestyle='dashed')
        plt.plot(bias_bx1, bias_by1, c='blue', linestyle='dashed')
        plt.plot(bias_tx2, bias_ty2, c='green', linestyle='dashed')
        plt.plot(bias_bx2, bias_by2, c='green', linestyle='dashed')
        plt.scatter(cobb1_pr, bias_cobb1, c='r',marker='x')
        plt.title('PT')
        plt.xlim((0,80))
        plt.ylim((-30,30))
        plt.xlabel('Measurement angle', color='#1C2833')
        plt.ylabel('Bias', color='#1C2833')
        plt.savefig('./imgs/cobb1_bias.jpg')
        plt.clf()
        plt.close()

        plt.plot(bias_x, bias_y, c='yellow', linestyle='dashed')
        plt.plot(bias_tx1, bias_ty1, c='blue', linestyle='dashed')
        plt.plot(bias_bx1, bias_by1, c='blue', linestyle='dashed')
        plt.plot(bias_tx2, bias_ty2, c='green', linestyle='dashed')
        plt.plot(bias_bx2, bias_by2, c='green', linestyle='dashed')
        plt.scatter(cobb2_pr, bias_cobb2, c='r',marker='x')
        plt.title('MT')
        plt.xlim((0,80))
        plt.ylim((-30,30))
        plt.xlabel('Measurement angle', color='#1C2833')
        plt.ylabel('Bias', color='#1C2833')
        plt.savefig('./imgs/cobb2_bias.jpg')
        plt.clf()
        plt.close()

        plt.plot(bias_x, bias_y, c='yellow', linestyle='dashed')
        plt.plot(bias_tx1, bias_ty1, c='blue', linestyle='dashed')
        plt.plot(bias_bx1, bias_by1, c='blue', linestyle='dashed')
        plt.plot(bias_tx2, bias_ty2, c='green', linestyle='dashed')
        plt.plot(bias_bx2, bias_by2, c='green', linestyle='dashed')
        plt.scatter(cobb3_pr, bias_cobb3, c='r',marker='x')
        plt.title('TL')
        plt.xlim((0,80))
        plt.ylim((-30,30))
        plt.xlabel('Measurement angle', color='#1C2833')
        plt.ylabel('Bias', color='#1C2833')
        plt.savefig('./imgs/cobb3_bias.jpg')
        plt.clf()
        plt.close()

