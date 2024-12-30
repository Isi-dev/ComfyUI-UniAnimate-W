'''
/* 
*Copyright (c) 2021, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
'''

import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
# import pynvml
import logging
import numpy as np
from PIL import Image
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from einops import rearrange
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel

from ...utils import transforms as data
from ..modules.config import cfg
from ...utils.seed import setup_seed
from ...utils.multi_port import find_free_port
from ...utils.assign_cfg import assign_signle_cfg
from ...utils.distributed import generalized_all_gather, all_reduce
from ...utils.video_op import save_i2vgen_video, save_t2vhigen_video_safe, save_video_multiple_conditions_not_gif_horizontal_3col
from ...tools.modules.autoencoder import get_first_stage_encoding
from ...utils.registry_class import INFER_ENGINE, MODEL2, EMBEDDER, AUTO_ENCODER, DIFFUSION
from copy import copy
import cv2


# @INFER_ENGINE.register_function()
def inference_animate_x_entrance_v2(seed, steps, useFirstFrame, reference_image, ref_pose, pose_sequence, original_driven_video_path, refpose_embeding_key, pose_embedding_key, frame_interval, max_frames, resolution, cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        return worker(0, seed, steps, useFirstFrame, reference_image, ref_pose, pose_sequence, original_driven_video_path, refpose_embeding_key, pose_embedding_key,frame_interval, max_frames, resolution, cfg, cfg_update)
    else:
        return mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg


def process_single_pose_embedding_tensor(pose_data):
    # Handle NumPy arrays directly
    if isinstance(pose_data, np.ndarray):
        bodies = pose_data[:18]  # Extract the first 18 rows (if not already done)
    else:
        raise TypeError(f"Expected pose_data to be a numpy.ndarray, got {type(pose_data)}")

    results = np.swapaxes(bodies, 0, 1)  # Swap axes for expected output
    # print(f"Processed embedding: {results}")
    return results

def process_single_pose_embedding_katong_tensor(pose_data, index):
    # Handle lists of NumPy arrays
    if isinstance(pose_data, list) and all(isinstance(body, np.ndarray) for body in pose_data):
        bodies = pose_data[index][:18]  # Extract the first 18 rows for the given index
    else:
        raise TypeError(f"Expected pose_data to be a list of numpy.ndarray, got {type(pose_data)}")

    results = np.swapaxes(bodies, 0, 1)  # Swap axes for expected output
    # print(f"Processed embedding2: {results}")
    return results


def load_video_frames(ref_image_tensor, ref_pose_tensor, pose_tensors, original_driven_video_path, refpose_embeding_key, pose_embedding_key,train_trans, vit_transforms, train_trans_pose, max_frames=32, frame_interval=1, resolution=[512, 768], get_first_frame=True, vit_resolution=[224, 224]):    
    pose_embedding_dim = 18
    
    for _ in range(5):
        try:
            num_poses = len(pose_tensors)
            numpyFrames = []
            numpyPoses = []

            original_driven_video_all = []
            original_driven_video_frame_all = []
            pose_embedding_all = []

            try:
                ref_pose_embedding = process_single_pose_embedding_tensor(refpose_embeding_key)
                # print("refpose_embeding_key successful!")

            except:
                print(f"Error processing refBody!")

            # print(f'i_frame is {ref_pose_embedding}')
            
            # Convert tensors to numpy arrays and prepare lists
            for i in range(num_poses):
                frame = ref_image_tensor.squeeze(0).cpu().numpy()  # Convert to numpy array
                # if i == 0:
                #     print(f'ref image is ({frame})')
                numpyFrames.append(frame)

                pose = pose_tensors[i].squeeze(0).cpu().numpy()  # Convert to numpy array
                numpyPoses.append(pose)

                original_driven_video_frame = original_driven_video_path[i].squeeze(0).cpu().numpy()
                original_driven_video_all.append(original_driven_video_frame)

                try:
                    pose_embedding = process_single_pose_embedding_katong_tensor(pose_embedding_key, i)
                    pose_embedding_all.append(pose_embedding) 
                    # print("pose_embedding_key successful!")
                except Exception as e:
                    print(f"Error processing pose_embedding_key! {e}")

            # Convert reference pose tensor to numpy array
            pose_ref = ref_pose_tensor.squeeze(0).cpu().numpy()  # Convert to numpy array

            # Sample max_frames poses for video generation
            stride = frame_interval
            total_frame_num = len(numpyFrames)
            cover_frame_num = (stride * (max_frames - 1) + 1)

            if total_frame_num < cover_frame_num:
                print(f'_total_frame_num ({total_frame_num}) is smaller than cover_frame_num ({cover_frame_num}), the sampled frame interval is changed')
                start_frame = 0
                end_frame = total_frame_num
                stride = max((total_frame_num - 1) // (max_frames - 1), 1)
                end_frame = stride * max_frames
            else:
                start_frame = 0
                end_frame = start_frame + cover_frame_num

            frame_list = []
            dwpose_list = []
            original_driven_video_list = []
            pose_embedding_list = []


            print(f'end_frame is ({end_frame})')

            for i_index in range(start_frame, end_frame, stride):
                if i_index < len(numpyFrames):  # Check index within bounds
                    i_frame = numpyFrames[i_index]
                    i_dwpose = numpyPoses[i_index]

                    # i_key = list(numpyFrames.keys())[i_index]

                    # Convert numpy arrays to PIL images
                    # i_frame = np.clip(i_frame, 0, 1)
                    # print("processing image frame")
                    i_frame = (i_frame - i_frame.min()) / (i_frame.max() - i_frame.min()) #Trying this in place of clip
                    i_frame = Image.fromarray((i_frame * 255).astype(np.uint8))
                    i_frame = i_frame.convert('RGB')
                    # i_dwpose = np.clip(i_dwpose, 0, 1)
                    # print("processing pose")
                    i_dwpose = (i_dwpose - i_dwpose.min()) / (i_dwpose.max() - i_dwpose.min()) #Trying this in place of clip
                    i_dwpose = Image.fromarray((i_dwpose * 255).astype(np.uint8))
                    i_dwpose = i_dwpose.convert('RGB')

                    i_original_driven_video = original_driven_video_all[i_index]
                    # print("processing video frame")
                    i_original_driven_video = (i_original_driven_video - i_original_driven_video.min()) / (i_original_driven_video.max() - i_original_driven_video.min()) #Trying this in place of clip
                    i_original_driven_video = Image.fromarray((i_original_driven_video * 255).astype(np.uint8))
                    i_original_driven_video = i_original_driven_video.convert('RGB')

                    # print("heading to embedding")

                    i_pose_embedding = pose_embedding_all[i_index]

                    # print("Done with embedding")

                    # if i_index == 0:
                    #     print(f'i_frame is ({np.array(i_frame)})')

                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)

                    original_driven_video_list.append(i_original_driven_video)

                    pose_embedding_list.append(i_pose_embedding)


            if frame_list:
                # random_ref_frame = np.clip(numpyFrames[0], 0, 1)
                # print("There's a frame list. We will proceed with further processing.")
                random_ref_frame = (numpyFrames[0] - numpyFrames[0].min()) / (numpyFrames[0].max() - numpyFrames[0].min()) #Trying this in place of clip
                random_ref_frame = Image.fromarray((random_ref_frame * 255).astype(np.uint8))
                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')
                # random_ref_dwpose = np.clip(pose_ref, 0, 1)
                random_ref_dwpose = (pose_ref - pose_ref.min()) / (pose_ref.max() - pose_ref.min()) #Trying this in place of clip
                random_ref_dwpose = Image.fromarray((random_ref_dwpose * 255).astype(np.uint8))
                if random_ref_dwpose.mode != 'RGB':
                    random_ref_dwpose = random_ref_dwpose.convert('RGB')

                # Apply transforms
                ref_frame = frame_list[0]
                vit_frame = vit_transforms(ref_frame)
                random_ref_frame_tmp = train_trans_pose(random_ref_frame)
                random_ref_dwpose_tmp = train_trans_pose(random_ref_dwpose)

                
                original_driven_video_data_tmp = torch.stack([vit_transforms(ss) for ss in original_driven_video_list], dim=0)

                
                ref_pose_embedding_tmp = torch.from_numpy(ref_pose_embedding)

                misc_data_tmp = torch.stack([train_trans_pose(ss) for ss in frame_list], dim=0)
                video_data_tmp = torch.stack([train_trans(ss) for ss in frame_list], dim=0)
                dwpose_data_tmp = torch.stack([train_trans_pose(ss) for ss in dwpose_list], dim=0)

                pose_embedding_tmp = torch.stack([torch.from_numpy(ss) for ss in pose_embedding_list], dim=0)

                # Initialize tensors
                video_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
                dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])

                original_driven_video_data = torch.zeros(max_frames, 3, 224, 224)

                pose_embedding = torch.zeros(max_frames, 2, pose_embedding_dim)
                ref_pose_embedding = torch.zeros(max_frames, 2, pose_embedding_dim)

                misc_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
                random_ref_frame_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
                random_ref_dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])

                # Copy data to tensors
                video_data[:len(frame_list), ...] = video_data_tmp
                misc_data[:len(frame_list), ...] = misc_data_tmp
                dwpose_data[:len(frame_list), ...] = dwpose_data_tmp

                original_driven_video_data[:len(frame_list), ...] = original_driven_video_data_tmp

                pose_embedding[:len(frame_list), ...] = pose_embedding_tmp

                random_ref_frame_data[:,...] = random_ref_frame_tmp
                random_ref_dwpose_data[:,...] = random_ref_dwpose_tmp

                ref_pose_embedding[:,...] = ref_pose_embedding_tmp

                return vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data, pose_embedding, ref_pose_embedding, original_driven_video_data

        except Exception as e:
            # logging.info(f'Error reading video frame: {e}')
            print(f'Error reading video frame: ({e})')
            continue
    
    return None, None, None, None, None, None, None, None, None  # Return default values if all attempts fail


def worker(gpu, seed, steps, useFirstFrame, reference_image, ref_pose, pose_sequence, original_driven_video_path, refpose_embeding_key, pose_embedding_key,frame_interval, max_frames, resolution, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    def is_libuv_supported():
        build_info = torch.__config__.show()
        return 'USE_LIBUV' in build_info
    
    try:
        if not is_libuv_supported():
            print("libuv is not supported, disabling USE_LIBUV")
            os.environ["USE_LIBUV"] = "0"
    except Exception as e:
        print(f"Unexpected error occured: {e}")
        os.environ["USE_LIBUV"] = "0"

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
            torch.backends.cudnn.benchmark = False
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    # log_dir = generalized_all_gather(cfg.log_dir)[0]
    # inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    # test_model = osp.basename(cfg.test_model).split('.')[0].split('_')[-1]
    
    # cfg.log_dir = osp.join(cfg.log_dir, '%s' % (inf_name))
    # os.makedirs(cfg.log_dir, exist_ok=True)
    # log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    # cfg.log_file = log_file
    # reload(logging)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='[%(asctime)s] %(levelname)s: %(message)s',
    #     handlers=[
    #         logging.FileHandler(filename=log_file),
    #         logging.StreamHandler(stream=sys.stdout)])
    # logging.info(cfg)
    # logging.info(f"Running UniAnimate inference on gpu {gpu}")
    print(f'Running Animate_X_v2 inference on gpu ({gpu})')
    cfg.resolution = resolution
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.Resize(resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    train_trans_pose = data.Compose([
        data.Resize(resolution),
        data.ToTensor(),
        ]
        )

    # Defines transformations for data to be fed into a Vision Transformer (ViT) model.
    vit_transforms = T.Compose([
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)]) 

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    with torch.no_grad():
        _, _, zero_y = clip_encoder(text="")
    

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters(): 
        param.requires_grad = False 
    autoencoder.cuda()
    
    # [Model] UNet 
    if "config" in cfg.UNet:
        cfg.UNet["config"] = cfg
    cfg.UNet["zero_y"] = zero_y
    if max_frames > 32:
        cfg.UNet["seq_len"] = max_frames+1
    model = MODEL2.build(cfg.UNet)
    # Here comes the UniAnimate model
    # inferences folder
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # tools folder
    parent_directory = os.path.dirname(current_directory)
    # Animate-x folder
    root_directory = os.path.dirname(parent_directory)
    unifiedModel = os.path.join(root_directory, 'checkpoints/animate-x_ckpt.pth')
    state_dict = torch.load(unifiedModel, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'step' in state_dict:
        resume_step = state_dict['step']
    else:
        resume_step = 0
    try:
        status = model.load_state_dict(state_dict, strict=False)
    except:

        for key in list(state_dict.keys()):
            if 'pose_embedding_before.pos_embed.pos_table' in key:  
                del state_dict[key]
        status = model.load_state_dict(state_dict, strict=False)
    # status = model.load_state_dict(state_dict, strict=True)
    # logging.info('Load model from {} with status {}'.format(unifiedModel, status))
    print(f'Load model from ({unifiedModel}) with status ({status})')
    model = model.to(gpu)
    model.eval()
    if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
        print("Avoiding DistributedDataParallel to reduce memory usage")
        model.to(torch.float16) 
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()


    # Where the input image and pose images come in
    test_list = cfg.test_list_path
    # num_videos = len(test_list)
    # logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    # test_list = [item for item in test_list for _ in range(cfg.round)]
    test_list = [item for _ in range(cfg.round) for item in test_list]
    
    # for idx, file_path in enumerate(test_list):

    # You can start inputs here for any user interface
    # Inputs will be ref_image_key, pose_seq_key, frame_interval, max_frames, resolution
    # cfg.frame_interval, ref_image_key, pose_seq_key = file_path[0], file_path[1], file_path[2]
    
    manual_seed = int(cfg.seed + cfg.rank)
    setup_seed(manual_seed)
    # logging.info(f"[{idx}]/[{len(test_list)}] Begin to sample {ref_image_key}, pose sequence from {pose_seq_key} init seed {manual_seed} ...")
    print(f"Seed: {manual_seed}")

    
    # initialize reference_image, pose_sequence, frame_interval, max_frames, resolution_x,
    vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data, pose_embedding, ref_pose_embedding, original_driven_video_data = load_video_frames(reference_image, ref_pose, pose_sequence, original_driven_video_path, refpose_embeding_key, pose_embedding_key,train_trans, vit_transforms, train_trans_pose, max_frames, frame_interval, resolution)
    # vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data = load_video_frames(reference_image, ref_pose, pose_sequence,train_trans, vit_transforms, train_trans_pose, max_frames, frame_interval, resolution)
    original_driven_video_data = torch.cat([vit_frame.unsqueeze(0), original_driven_video_data], 0)
    
    misc_data = misc_data.unsqueeze(0).to(gpu)
    vit_frame = vit_frame.unsqueeze(0).to(gpu)
    dwpose_data = dwpose_data.unsqueeze(0).to(gpu) 
    original_driven_video_data = original_driven_video_data.unsqueeze(0).to(gpu)
    random_ref_frame_data = random_ref_frame_data.unsqueeze(0).to(gpu)
    random_ref_dwpose_data = random_ref_dwpose_data.unsqueeze(0).to(gpu)

    pose_embedding = pose_embedding.unsqueeze(0).to(gpu)
    ref_pose_embedding = ref_pose_embedding[0:1].unsqueeze(0).to(gpu)

    pose_embedding = torch.cat([ref_pose_embedding, pose_embedding], dim = 1)

    

    ### save for visualization
    misc_backups = copy(misc_data)
    frames_num = misc_data.shape[1]
    misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
    mv_data_video = []
    

    ### local image (first frame)
    image_local = []
    if 'local_image' in cfg.video_compositions:
        frames_num = misc_data.shape[1]
        bs_vd_local = misc_data.shape[0]
        # create a repeated version of the first frame across all frames and assign to image_local
        image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
        image_local_clone = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
        image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
        if hasattr(cfg, "latent_local_image") and cfg.latent_local_image:
            with torch.no_grad(): # Disable gradient calculation
                temporal_length = frames_num
                # The encoder compresses the input data into a lower-dimensional latent representation, often called a "latent vector" or "encoding."
                encoder_posterior = autoencoder.encode(video_data[:,0])
                local_image_data = get_first_stage_encoding(encoder_posterior).detach() #use without affecting the gradients of the original model
                image_local = local_image_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]
        

    
    ### encode the video_data
    # bs_vd = misc_data.shape[0]
    misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')
    # misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)
    

    with torch.no_grad():
        
        random_ref_frame = []
        if 'randomref' in cfg.video_compositions:
            random_ref_frame_clone = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')
            if hasattr(cfg, "latent_random_ref") and cfg.latent_random_ref:
                
                temporal_length = random_ref_frame_data.shape[1]
                encoder_posterior = autoencoder.encode(random_ref_frame_data[:,0].sub(0.5).div_(0.5))
                random_ref_frame_data = get_first_stage_encoding(encoder_posterior).detach()
                random_ref_frame_data = random_ref_frame_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

            random_ref_frame = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')


        if 'dwpose' in cfg.video_compositions:
            bs_vd_local = dwpose_data.shape[0]
            dwpose_data_clone = rearrange(dwpose_data.clone(), 'b f c h w -> b c f h w', b = bs_vd_local)
            if 'randomref_pose' in cfg.video_compositions:
                dwpose_data = torch.cat([random_ref_dwpose_data[:,:1], dwpose_data], dim=1)
            dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)

        
        y_visual = []
        if 'image' in cfg.video_compositions:
            with torch.no_grad():
                vit_frame = vit_frame.squeeze(1)
                y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                y_visual0 = y_visual.clone()


        batch_size, seq_len = original_driven_video_data.shape[0], original_driven_video_data.shape[1]

        original_driven_video_data = original_driven_video_data.reshape(batch_size*seq_len,3,224,224)
        original_driven_video_data_embedding = clip_encoder.encode_image(original_driven_video_data).unsqueeze(1) # [60, 1024]
        
        # print("original_driven_video_data_embedding.shape: ", original_driven_video_data_embedding.shape)
        original_driven_video_data_embedding = original_driven_video_data_embedding.clone()
    
    # print(torch.get_default_dtype())

    with amp.autocast(enabled=True):
        # pynvml.nvmlInit()
        # handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        cur_seed = torch.initial_seed()
        # logging.info(f"Current seed {cur_seed} ...")

        print(f"Number of frames to denoise: {frames_num}")
        noise = torch.randn([1, 4, frames_num, int(resolution[1]/cfg.scale), int(resolution[0]/cfg.scale)])
        noise = noise.to(gpu)      
        # print(f"noise: {noise.shape}")

        # add a noise prior
        noise = diffusion.q_sample(random_ref_frame.clone(), getattr(cfg, "noise_prior_value", 949), noise=noise)
        
        
        if hasattr(cfg.Diffusion, "noise_strength"):
            b, c, f, _, _= noise.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
            noise = noise + cfg.Diffusion.noise_strength * offset_noise
            # print(f"offset_noise dtype: {offset_noise.dtype}")
            # print(f' offset_noise is ({offset_noise})')

    
        
        

        # construct model inputs (CFG)
        full_model_kwargs=[{
                                    'y': None,
                                    'pose_embeddings': [pose_embedding, original_driven_video_data_embedding],
                                    "local_image": None if len(image_local) == 0 else image_local[:],
                                    'image': None if len(y_visual) == 0 else y_visual0[:],
                                    'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                                    'randomref': None if len(random_ref_frame) == 0 else random_ref_frame[:],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    "pose_embeddings": None,
                                    }]

        # for visualization
        full_model_kwargs_vis =[{
                                    'y': None,
                                    "local_image": None if len(image_local) == 0 else image_local_clone[:],
                                    'image': None,
                                    'pose_embeddings': [pose_embedding, original_driven_video_data_embedding],
                                    'dwpose': None if len(dwpose_data_clone) == 0 else dwpose_data_clone[:],
                                    'randomref': None if len(random_ref_frame) == 0 else random_ref_frame_clone[:, :3],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    "pose_embeddings": None, 
                                    }]

        
        partial_keys = [
                ['image', 'randomref', "dwpose","pose_embeddings"],
                # ['image', 'randomref', "dwpose"],
            ]

        if useFirstFrame:
            partial_keys = [
                ['image', 'local_image', "dwpose","pose_embeddings"],
                # ['image', 'local_image', "dwpose"],
            ]
            print('Using First Frame Conditioning!')

       
        for partial_keys_one in partial_keys:
            model_kwargs_one = prepare_model_kwargs(partial_keys = partial_keys_one,
                                full_model_kwargs = full_model_kwargs,
                                use_fps_condition = cfg.use_fps_condition)
            model_kwargs_one_vis = prepare_model_kwargs(partial_keys = partial_keys_one,
                                full_model_kwargs = full_model_kwargs_vis,
                                use_fps_condition = cfg.use_fps_condition)
            noise_one = noise
            
            if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                clip_encoder.cpu() # add this line
                del clip_encoder  # Delete this object to free memory
                autoencoder.cpu() # add this line
                torch.cuda.empty_cache() # add this line
                import gc
                gc.collect()

            # print(f' noise_one is ({noise_one})')
            # print(f"noise: {noise.shape}")
                
            video_data = diffusion.ddim_sample_loop(
                noise=noise_one,
                model=model.eval(), 
                model_kwargs=model_kwargs_one,
                guide_scale=cfg.guide_scale,
                ddim_timesteps=steps,
                eta=0.0)
            
            # print(f"video_data dtype: {video_data.dtype}")
            
            if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                # if run forward of  autoencoder or clip_encoder second times, load them again
                # clip_encoder.cuda()
                del diffusion
                torch.cuda.empty_cache()
                gc.collect()
                autoencoder.cuda()

            video_data = 1. / cfg.scale_factor * video_data 
            video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
            chunk_size = min(cfg.decoder_bs, video_data.shape[0])
            video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
            decode_data = []
            for vd_data in video_data_list:
                gen_frames = autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
            video_data = torch.cat(decode_data, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size).float()


            del model_kwargs_one_vis[0][list(model_kwargs_one_vis[0].keys())[0]]
            del model_kwargs_one_vis[1][list(model_kwargs_one_vis[1].keys())[0]]

            video_data = extract_image_tensors(video_data.cpu(), cfg.mean, cfg.std)

            
            # synchronize to finish some processes
            if not cfg.debug:
                torch.cuda.synchronize()
                dist.barrier()

            return video_data

@torch.no_grad()
def extract_image_tensors(video_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Unnormalize the video tensor
    mean = torch.tensor(mean, device=video_tensor.device).view(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(std, device=video_tensor.device).view(1, -1, 1, 1, 1)    # ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  # unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    images = rearrange(video_tensor, 'b c f h w -> b f h w c')
    images = images.squeeze(0)
    images_t = []
    for img in images:
        img_array = np.array(img)  # Convert PIL Image to numpy array
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor and CHW format
        img_tensor = img_tensor.permute(0, 2, 3, 1)
        images_t.append(img_tensor)

    # logging.info('Images data extracted!')
    print('Inference completed!')
    images_t = torch.cat(images_t, dim=0)
    return images_t

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):   
    if use_fps_condition is True:
        partial_keys.append('fps')
    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]
    return partial_model_kwargs
