import torch

from .utils.config import Config
from .tools.inferences import inference_unianimate_entrance
from . import run_align_pose

# from tools import *



class UniAnimateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 25, "max": 50, "step": 1}),
                "useFirstFrame": ("BOOLEAN", { "default": False }),
                "reference_image": ("IMAGE",),  # single image
                "ref_pose": ("IMAGE",),  # single image
                "pose_sequence": ("IMAGE",),   # Batch of pose images
                "frame_interval": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "max_frames": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1}),
                "resolution_x": ("INT", {"default": 512, "min": 512, "max": 768, "step": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, steps, useFirstFrame, reference_image, ref_pose, pose_sequence, frame_interval, max_frames, resolution_x):
        cfg_update = Config(load=True)
        resolution_y = 768
        if resolution_x == 768:
            resolution_y = 1216
        resolution = [resolution_x, resolution_y]
        print("Ready for inference.")
        
        # print(f"image is: {reference_image}")
        
        frames = inference_unianimate_entrance(steps, useFirstFrame, reference_image, ref_pose, pose_sequence, frame_interval, max_frames, resolution, cfg_update=cfg_update.cfg_dict)
        mask_template = torch.zeros((1, resolution_y, resolution_x), dtype=torch.float32)
        masks = [mask_template.clone() for _ in range(len(pose_sequence))]
        masks = torch.cat(masks, dim=0)
        return (frames, masks)

class Gen_align_pose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),  # single image
                "video": ("IMAGE",),   # video
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, reference_image, video):    
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available")
        poses, refPose = run_align_pose.mp_main(reference_image, video)
        return (refPose, poses)



NODE_CLASS_MAPPINGS = {
    "UniAnimateImage" : UniAnimateImage,
    "Gen_align_pose" : Gen_align_pose,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniAnimateImage" :"Animate image with UniAnimate",
    "Gen_align_pose" :"Align & Generate poses for UniAnimate",
    
}