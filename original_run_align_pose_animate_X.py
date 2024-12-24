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


def mp_main(reference_image, video):
    
    logger.info(f"There are {video.size(0)} frames for extracting poses")

    logger.info('LOAD: DW Pose Model')
    dwpose_model = DWposeDetector()  
    
    results_vis = []
    num_frames = video.size(0)

    # frames_numpy = video.permute(0, 2, 3, 1).cpu().numpy()

    # bodies = []
    # body_indices = []
    # hands = []
    # faces = []
    # fps = 30



    # refPose_data = {
    #     "bodies": [],
    #     "body_indices": [],
    #     "faces": [],
    #     "hands": [],
    #     "size": [],
    #     "fps": 30,
    # }
    size = (512, 768, 3)


    for i in range(num_frames):
        logger.info(f"Processing frame {i + 1}/{num_frames}")
        frame = video[i].permute(0, 1, 2).cpu().numpy() # Convert to HWC format and numpy array
        frame = ((frame - frame.min()) / (frame.max() - frame.min()))*255 
        frame = np.flip(frame, axis=2)  
        pose = dw_func(i, frame, dwpose_model)
        size = frame.shape # (1216, 832, 3)

        # bodies.append(pose['bodies']['candidate'][:18])
        # body_indices.append(pose['bodies']['subset'][0][:18])
        # faces.append(pose['faces'][0])
        # hands.append(pose['hands'])

        results_vis.append(pose)

    logger.info(f'All frames have been processed.')
    print(len(results_vis))

    # pose_data = {}
    # pose_data['bodies'] = np.array(bodies)
    # pose_data['body_indices'] = np.array(body_indices)
    # pose_data['faces'] = np.array(faces)
    # pose_data['hands'] = np.array(hands)
    # pose_data['size'] = size
    # pose_data['fps'] = fps

    # Process the reference image
    ref_frame = reference_image.squeeze(0).cpu().numpy()  # Convert to HWC format and numpy array
    ref_frame = ((ref_frame - ref_frame.min()) / (ref_frame.max() - ref_frame.min()))*255
    ref_frame = np.flip(ref_frame, axis=2)
    pose_ref = dw_func(-1, ref_frame, dwpose_model)
    ref_size = ref_frame.shape


    # refPose_data["bodies"]=pose['bodies']['candidate'][:18]
    # refPose_data["body_indices"]=pose['bodies']['subset'][0][:18]
    # refPose_data["faces"]=pose['faces'][0]
    # refPose_data["hands"]=pose['hands']
    # refPose_data["size"] = ref_frame.shape
    # refPose_data['fps'] = fps


    dwpose_images = []
    (H,W,_) = size

    for i in range(len(results_vis)):
        dwpose_woface, _ = draw_pose(results_vis[i], H, W)
        dwpose_tensor = torch.from_numpy(dwpose_woface).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor and CHW format
        dwpose_tensor = dwpose_tensor.permute(0, 2, 3, 1)
        dwpose_images.append(dwpose_tensor)
    
    dwpose_images = torch.cat(dwpose_images, dim=0)

    (H,W,_) = ref_size
    dwpose_woface_ref, _ = draw_pose(pose_ref, H, W)
    dwpose_ref_tensor = torch.from_numpy(dwpose_woface_ref).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor and CHW format
    dwpose_ref_tensor = dwpose_ref_tensor.permute(0, 2, 3, 1)


    # print(f'The type of the pose from run_align_pose is currently of the form : {type(dwpose_ref_tensor)} ')
    # print(f'The content of the pose from run_align_pose is currently: {dwpose_ref_tensor} ')
    
    return dwpose_images, dwpose_ref_tensor


logger = get_logger('dw pose extraction')

