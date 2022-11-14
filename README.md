# Curvature-of-Cervical-Spine-Estimation

This repository includes the official implementation of the paper:

HTN: Hybrid Transformer Network for Curvature of Cervical Spine Estimation

(State: Review in Second Round, On Applied Sciences, MDPI)

Authors and affiliations:
Yifan Yao 1, 2, Jiuqing Dong 2, Wenjun Yu 1, and Yongbin Gao 1,*
1	International Joint Research Lab of Intelligent Perception and Control, Shanghai University of Engineering Science, No. 333 Longteng Road, Shanghai 201620, P.R China
2	Division of Computer Science and Engineering, Jeonbuk National University, Jeonju 54896, Korea

The code and our model will be released in two weeks.

# Install
Clone repo and install related library in a Python>=3.6.0 environment, including PyTorch>=1.5.

# Training
python main.py --data_dir dataPath --num_epoch 100 --batch_size 2 --dataset spinal --phase train

# Testing and visualization
python main.py --resume weightPath --data_dir dataPath --dataset spinal  --phase test

# Evaluation and result scatter plot
python main.py --resume weightPath --data_dir dataPath --dataset spinal  --phase eval

# Note
if you have any problems, please contact us.
