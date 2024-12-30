# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import json
import copy
import torch
import random
import argparse
import shutil
import tempfile
import subprocess
import numpy as np
import math

import torch.multiprocessing as mp
import torch.distributed as dist
# import pickle
import logging
# from io import BytesIO
# import oss2 as oss
# import os.path as osp

import sys
from .dwpose import util
from .dwpose.wholebody import Wholebody


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger

class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        # print(f'The shape of the image should be in HWC format but it is currently: {oriImg.shape} ')
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) 
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) 
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_body_and_foot(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas

def dw_func(_id, frame, dwpose_model, dwpose_woface_folder='tmp_dwpose_wo_face', dwpose_withface_folder='tmp_dwpose_with_face'):
    
    # frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    pose = dwpose_model(frame)

    return pose


def mp_main(dontAlignPose, reference_image, video):
    
    logger.info(f"There are {video.size(0)} frames for extracting poses")

    logger.info('LOAD: DW Pose Model')
    dwpose_model = DWposeDetector()  
    
    results_vis = []
    num_frames = video.size(0)

    frames_numpy = video.permute(0, 2, 3, 1).cpu().numpy()

    bodiesA = []
    body_indices = []
    hands = []
    faces = []
    fps = 30



    refPose_data = {
        "bodies": [],
        "body_indices": [],
        "faces": [],
        "hands": [],
        "size": [],
        "fps": 30,
    }


    for i in range(num_frames):
        logger.info(f"Processing frame {i + 1}/{num_frames}")
        frame = video[i].permute(0, 1, 2).cpu().numpy() # Convert to HWC format and numpy array
        frame = ((frame - frame.min()) / (frame.max() - frame.min()))*255 
        frame = np.flip(frame, axis=2)  
        pose = dw_func(i, frame, dwpose_model)
        size = frame.shape # (1216, 832, 3)

        bodiesA.append(pose['bodies']['candidate'][:18])
        body_indices.append(pose['bodies']['subset'][0][:18])
        faces.append(pose['faces'][0])
        hands.append(pose['hands'])

        results_vis.append(pose)

    logger.info(f'All frames have been processed.')
    print(len(results_vis))

    pose_data = {}
    pose_data['bodies'] = np.array(bodiesA)
    pose_data['body_indices'] = np.array(body_indices)
    pose_data['faces'] = np.array(faces)
    pose_data['hands'] = np.array(hands)
    pose_data['size'] = size
    pose_data['fps'] = fps

    vid_not_aligned = copy.deepcopy(results_vis)

    # Process the reference image
    ref_frame = reference_image.squeeze(0).cpu().numpy()  # Convert to HWC format and numpy array
    ref_frame = ((ref_frame - ref_frame.min()) / (ref_frame.max() - ref_frame.min()))*255
    ref_frame = np.flip(ref_frame, axis=2)
    pose_ref = dw_func(-1, ref_frame, dwpose_model)

    refBody = pose_ref['bodies']['candidate'][:18]
    refPose_data["bodies"]=pose_ref['bodies']['candidate'][:18]
    refPose_data["body_indices"]=pose_ref['bodies']['subset'][0][:18]
    refPose_data["faces"]=pose_ref['faces'][0]
    refPose_data["hands"]=pose_ref['hands']
    refPose_data["size"] = ref_frame.shape
    refPose_data['fps'] = fps

    refBody = np.array(refBody)  # Ensure refBody is a NumPy array
    bodiesA = [np.array(body) for body in bodiesA]

    # print(f"bodies looks like {bodiesA}")

    # if isinstance(bodiesA, list) and all(isinstance(body, np.ndarray) for body in bodiesA):
    #     bodiesB = bodiesA[0][:18]  # Extract the first 18 rows for the given index
    # else:
    #     raise TypeError(f"Expected pose_data to be a list of numpy.ndarray, got {type(bodiesA)}")

    # print(f"refBody : {refBody}")
    # print(f"bodies : {bodies}")

    # if not isinstance(refBody, dict):
    #     raise TypeError(f"Expected refBody to be a dict, got {type(refBody)}")

    # if 'bodies' not in refBody or 'candidate' not in refBody['bodies']:
    #     print(f"refBody keys: {refBody.keys()}")
    #     raise KeyError("`refBody` does not contain the required key: 'bodies' or 'candidate'")
    
    # if not isinstance(bodies, dict):
    #     raise TypeError(f"Expected bodies to be a dict, got {type(bodies)}")

    # if 'bodies' not in bodies or 'candidate' not in bodies['bodies']:
    #     print(f"bodies keys: {bodies.keys()}")
    #     raise KeyError("`bodies` does not contain the required key: 'bodies' or 'candidate'")

    # print(f'The content of the image is currently: {pose_ref} ')




    bodies = results_vis[0]['bodies']
    faces = results_vis[0]['faces']
    hands = results_vis[0]['hands']
    candidate = bodies['candidate']

    ref_bodies = pose_ref['bodies']
    ref_faces = pose_ref['faces']
    ref_hands = pose_ref['hands']
    ref_candidate = ref_bodies['candidate']


    ref_2_x = ref_candidate[2][0]
    ref_2_y = ref_candidate[2][1]
    ref_5_x = ref_candidate[5][0]
    ref_5_y = ref_candidate[5][1]
    ref_8_x = ref_candidate[8][0]
    ref_8_y = ref_candidate[8][1]
    ref_11_x = ref_candidate[11][0]
    ref_11_y = ref_candidate[11][1]
    ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
    ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

    zero_2_x = candidate[2][0]
    zero_2_y = candidate[2][1]
    zero_5_x = candidate[5][0]
    zero_5_y = candidate[5][1]
    zero_8_x = candidate[8][0]
    zero_8_y = candidate[8][1]
    zero_11_x = candidate[11][0]
    zero_11_y = candidate[11][1]
    zero_center1 = 0.5*(candidate[2]+candidate[5])
    zero_center2 = 0.5*(candidate[8]+candidate[11])

    x_ratio, y_ratio = 1, 1
    if (zero_5_x-zero_2_x) > 0 :
        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x)
    if (zero_center2[1]-zero_center1[1]) > 0 :
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1])

    results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
    results_vis[0]['bodies']['candidate'][:,1] *= y_ratio
    results_vis[0]['faces'][:,:,0] *= x_ratio
    results_vis[0]['faces'][:,:,1] *= y_ratio
    results_vis[0]['hands'][:,:,0] *= x_ratio
    results_vis[0]['hands'][:,:,1] *= y_ratio
    
    ########neck########
    neck_ratio = 1
    l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
    if l_neck_0 != 0:
        neck_ratio = l_neck_ref / l_neck_0

    x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
    y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

    results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck
    
    ########shoulder2########
    shoulder2_ratio = 1
    l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5
    if l_shoulder2_0 != 0:
        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

    x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
    y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

    results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
    results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
    results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
    results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
    results_vis[0]['hands'][1,:,1] += y_offset_shoulder2

    ########shoulder5########
    shoulder5_ratio  = 1
    l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5
    if l_shoulder5_0 != 0:
        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

    x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
    y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

    results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
    results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
    results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
    results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
    results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

    ########arm3########
    arm3_ratio = 1
    l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
    l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5
    if l_arm3_0 != 0:
        arm3_ratio = l_arm3_ref / l_arm3_0

    x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
    y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

    results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
    results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
    results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
    results_vis[0]['hands'][1,:,0] += x_offset_arm3
    results_vis[0]['hands'][1,:,1] += y_offset_arm3

    ########arm4########
    arm4_ratio = 1
    l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
    l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5
    if l_arm4_0 != 0:
        arm4_ratio = l_arm4_ref / l_arm4_0

    x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
    y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

    results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
    results_vis[0]['hands'][1,:,0] += x_offset_arm4
    results_vis[0]['hands'][1,:,1] += y_offset_arm4

    ########arm6########
    arm6_ratio = 1
    l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
    l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5
    if l_arm6_0 != 0:
        arm6_ratio = l_arm6_ref / l_arm6_0

    x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
    y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

    results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
    results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
    results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
    results_vis[0]['hands'][0,:,0] += x_offset_arm6
    results_vis[0]['hands'][0,:,1] += y_offset_arm6

    ########arm7########
    arm7_ratio = 1
    l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
    l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5
    if l_arm7_0 != 0:
        arm7_ratio = l_arm7_ref / l_arm7_0

    x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
    y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

    results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
    results_vis[0]['hands'][0,:,0] += x_offset_arm7
    results_vis[0]['hands'][0,:,1] += y_offset_arm7

    ########head14########
    l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
    l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

    head14_ratio = l_head14_ref / l_head14_0

    x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
    y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

    results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
    results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
    results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14

    ########head15########
    l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
    l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

    head15_ratio = l_head15_ref / l_head15_0

    x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
    y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

    results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
    results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
    results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

    ########head16########
    l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
    l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

    head16_ratio = l_head16_ref / l_head16_0

    x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
    y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

    results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16

    ########head17########
    l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
    l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

    head17_ratio = l_head17_ref / l_head17_0

    x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
    y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

    results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17
    
    ########MovingAverage########
    
    ########left leg########
    ll1_ratio = 1
    l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
    l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
    if l_ll1_0 != 0 :
        ll1_ratio = l_ll1_ref / l_ll1_0

    x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
    y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

    results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
    results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
    results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

    l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
    l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
    ll2_ratio = l_ll2_ref / l_ll2_0

    x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
    y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

    results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
    results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
    results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
    results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2

    ########right leg########
    rl1_ratio = 1
    l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
    l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
    if l_rl1_0 != 0:
        rl1_ratio = l_rl1_ref / l_rl1_0

    x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
    y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

    results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
    results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
    results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1

    l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
    l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
    rl2_ratio = l_rl2_ref / l_rl2_0

    x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
    y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

    results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
    results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
    results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
    results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2

    offset = ref_candidate[1] - results_vis[0]['bodies']['candidate'][1]

    results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]
    results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
    results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]

    for i in range(1, len(results_vis)):
        results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[i]['faces'][:,:,0] *= x_ratio
        results_vis[i]['faces'][:,:,1] *= y_ratio
        results_vis[i]['hands'][:,:,0] *= x_ratio
        results_vis[i]['hands'][:,:,1] *= y_ratio

        ########neck########
        x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
        y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

        results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

        ########shoulder2########
        

        x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

        results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

        ########shoulder5########

        x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

        results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

        ########arm3########

        x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

        results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[i]['hands'][1,:,0] += x_offset_arm3
        results_vis[i]['hands'][1,:,1] += y_offset_arm3

        ########arm4########

        x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

        results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[i]['hands'][1,:,0] += x_offset_arm4
        results_vis[i]['hands'][1,:,1] += y_offset_arm4

        ########arm6########

        x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

        results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[i]['hands'][0,:,0] += x_offset_arm6
        results_vis[i]['hands'][0,:,1] += y_offset_arm6

        ########arm7########

        x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

        results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[i]['hands'][0,:,0] += x_offset_arm7
        results_vis[i]['hands'][0,:,1] += y_offset_arm7

        ########head14########

        x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
        y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

        results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

        ########head15########

        x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
        y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

        results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

        ########head16########

        x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
        y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

        results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

        ########head17########
        x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
        y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

        results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

        # ########MovingAverage########

        ########left leg########
        x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

        results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
        results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
        results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1



        x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

        results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
        results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
        results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

        ########right leg########

        x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

        results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
        results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
        results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1


        x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

        results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
        results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
        results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

        results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]


    dwpose_images = []

    for i in range(len(results_vis)):
        
        if dontAlignPose:
            dwpose_woface, _ = draw_pose(vid_not_aligned[i], H=768, W=512)
        else:
            dwpose_woface, _ = draw_pose(results_vis[i], H=768, W=512)
        dwpose_tensor = torch.from_numpy(dwpose_woface).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor and CHW format
        dwpose_tensor = dwpose_tensor.permute(0, 2, 3, 1)
        dwpose_images.append(dwpose_tensor)
    
    dwpose_images = torch.cat(dwpose_images, dim=0)

    dwpose_woface_ref, _ = draw_pose(pose_ref, H=768, W=512)
    dwpose_ref_tensor = torch.from_numpy(dwpose_woface_ref).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor and CHW format
    dwpose_ref_tensor = dwpose_ref_tensor.permute(0, 2, 3, 1)


    # print(f'The type of the pose from run_align_pose is currently of the form : {type(dwpose_ref_tensor)} ')
    # print(f'The content of the pose from run_align_pose is currently: {dwpose_ref_tensor} ')
    
    return dwpose_images, dwpose_ref_tensor, bodiesA, refBody


logger = get_logger('dw pose extraction')

