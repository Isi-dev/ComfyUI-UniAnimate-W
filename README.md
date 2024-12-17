<!-- main documents -->


<div align="center">

<img src="gif.gif" />

## ComfyUi-windows implementation for the image animation project -> UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation

[🎨 UniAnimate Project Page](https://unianimate.github.io/)

</div>

## Updates

17/12/2024: Released a [video](https://youtu.be/OKFf8J-eMIc) comparing Animate_X with UniAnimate 

16/12/2024: Added three nodes and associated workflows for Animate_X: `Repose image with Animate_X` for img2img pose transfer, `Animate image with Animate_X` for img2vid generation, and `Animate image with Animate_X_Long` for long video generation. 
Install or Update this repository with the ComfyUI Manager to get these nodes. You will need to download the `animate-x_ckpt.pth` model and place it in 'ComfyUI-UniAnimate-W/checkpoints/' folder to use these nodes. The other four checkpoints required are the same as those used by UniAnimate. 
You can download the checkpoints here:https://huggingface.co/Shuaishuai0219/Animate-X/tree/main
The code for Animate_X is almost the same as that for UniAnimate, so I decided to include the Animate_X nodes here rather than creating a new repository. The requirements are also the same.
You can visit the Animate_X repo via this link: https://github.com/antgroup/animate-x



09/09/2024: Released a [video](https://youtu.be/Ne-DSBhfg8A) on using the two new UniAnimate nodes for best results 

07/09/2024: Added two nodes: `Animate image with UniAnimate_Long` for long video generation, and `Repose image with UniAnimate` for img2img pose transfer


## Getting Started

The ComfyUI nodes created are `Align & Generate poses for UniAnimate` & `Animate image with UniAnimate`    

I used a ComfyUI_windows_portable to test the nodes in a Windows 10 OS with 16GB RAM & 12GB VRAM Nvidia Graphics Card

Download or clone this repository and place it in ComfyUI_windows_portable\ComfyUI\custom_nodes\. Or install via the ComfyUI Manager by searching for

```
UniAnimate Nodes for ComfyUI
```

You will need python>=3.9 in your ComfyUI Environment.
I tested the project with the following pytorch versions which you can install as follows

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

``` 

Or

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

```

If not installed, then:


```
pip install opencv-python
pip install pytorch_lightning
pip install lightning_utilities 
pip install lightning_fabric
pip install torchmetrics
pip install xFormers = 0.0.20 or copy torch 2.0.1 and supporting libraries and xFormer from A1111 and place in your Environment\Lib\site-packages (or) pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118 (for pytorch==2.3.1)
pip install oss2
pip install einops
pip install args
pip install onnxruntime-gpu==1.13.1
pip install modelscope

```


Download the required models (Around 14GB) after installing modelscope :

```
python modeldownloader.py

```

After downloading all the models, move them manually from 'checkpoints/iic/unianimate/' to the 'checkpoints' directory
Or move them via your command line interface:

```
python mv ./checkpoints/iic/unianimate/* ./checkpoints/

```

All the models should be in the  '\Path-to-UniAnimate-W\checkpoints' folder as follows:


```
./checkpoints/
|---- dw-ll_ucoco_384.onnx
|---- open_clip_pytorch_model.bin
|---- unianimate_16f_32f_non_ema_223000.pth 
|---- v2-1_512-ema-pruned.ckpt
└---- yolox_l.onnx

```

You can now upload the workflow in your '\Path-to-UniAnimate-W\' folder which is titled 'UniAnimateImg2Vid.json', install missing custom nodes with the ComfyUI Manager if necessary, upload a picture & video (You can use those in the 'assets' folder), and run!


**<font color=red> Note </font>**:

- > In the 'Align & Generate poses for UniAnimate' node, The first frame of the target pose sequence is used to calculate the scale coefficient for aligning the pose sequence with the reference image. If this frame includes the entire face and full-body pose (hands and feet), it will result in more accurate estimations and better video generation results.

- > To run the `Animate image with UniAnimate` node, **~12GB** of GPU memory will be used. If your GPU has less memory, you can reduce the max_frames value from 32 to 24, 16, or 8.

- > You can also generate a video first, and then upload the last frame of the video as a pic to generate the next frames with `useFirstFrame` set to true in the `Align & Generate poses for UniAnimate` node.

- > Generating 32 frames of video with a resolution of [512, 768] usually takes about 7 minutes.

You can also change the pose of an image to that of another image as shown below.

<div align="center">
<img src="repose.png" />

You can watch a video on the basic workflow [here](https://youtu.be/vR8EHoAQziI) 
<br></br>
You can watch a video on the Installation [here](https://youtu.be/NFnhELV4bG0)

</div>

## Support
If you find this project helpful, you can support me here:  
[![Buy Me a Coffee](https://img.shields.io/badge/Support-Buy%20Me%20a%20Coffee-orange?style=flat-square&logo=buy-me-a-coffee)](https://buymeacoffee.com/isiomo)

## Disclaimer
 
I am not responsible for any user-generated content. Users are fully responsible for their actions when using these nodes and the generative model. Neither I nor the contributors to the UniAnimate project have any legal affiliation with or accountability for users' behaviors. It is crucial to use these nodes and the generative model responsibly, following both ethical and legal standards.
