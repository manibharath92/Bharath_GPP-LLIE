import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import glob
import os
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2
from utils import util
from torchvision.utils import save_image
from download import load_model
from torch.nn import functional as F
import natsort
from torchvision.transforms import ToTensor
import cv2
import numpy as np

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))
def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)
def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def main(inp_dir):

    lr_dir = os.path.join(inp_dir, 'low')
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    local_prior_dir = os.path.join(inp_dir, 'local_prior')

    out_dir = os.path.join(inp_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.jpeg'))
    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    device = torch.device('cuda:0')
    state_dict = torch.load('/kaggle/working/BharathPreTrainedDS/weight_lol.pth')

    model = DiT_incontext_revise()
    model.load_state_dict(state_dict['dit'], strict=True)
    model = model.to(device)

    vae = AutoencoderKL()
    vae.load_state_dict(state_dict['vae'], strict=True)
    vae = vae.to(device)

    cond_lq = CondEncoder()
    cond_lq.load_state_dict(state_dict['cond'], strict=True)
    cond_lq = cond_lq.to(device)

    second_decoder = Decoder2()
    second_decoder.load_state_dict(state_dict['second_decoder'], strict=True)
    second_decoder = second_decoder.to(device)

    model.eval()
    diffusion_val = create_diffusion(str(25))  # number of sample steps

    to_tensor = ToTensor()

    for lr_path, global_path, local_path, test_index in zip(lr_paths, global_prior_paths, local_prior_paths, range(len(lr_paths))):
        print(f"Processing image {test_index + 1}: {os.path.basename(lr_path)}")

        # Load and Resize to a manageable resolution (e.g., 512x512)
        img_bgr = cv2.imread(lr_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512)) # Force resolution to 512
        
        y = to_tensor(img_resized).unsqueeze(0).to(device)
        global_prior = torch.load(global_path, map_location=device)
        local_prior = torch.load(local_path, map_location=device)

        with torch.no_grad():
            # Memory clearing before heavy lifting
            torch.cuda.empty_cache() 
            
            y_feat, enc_feat = cond_lq(y, True)
            
            # Use the resized height/width for latent size
            b, c, h, w = y.shape
            z = torch.randn(1, 3, h // 4, w // 4, device=device)
            print(f"Latent z shape: {z.shape} -> tokens: {z.shape[2] * z.shape[3]}")
            print(f"Global prior shape: {global_prior.shape}")
            print(f"Local prior (q_map) shape: {local_prior.shape}")
            model_kwargs = dict(y=y_feat, vis=global_prior, q_map=local_prior)

            samples = diffusion_val.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=True, device=device
            )

            dec_feat = vae.decode(samples, mid_feat=True)
            sr = second_decoder(samples, dec_feat, enc_feat)
                  
        save_img_path = os.path.join(out_dir, os.path.basename(lr_path))                   
        save_image(sr, save_img_path)
        print(f"Successfully saved: {save_img_path}")


if __name__ == "__main__":

    input_dir = '/kaggle/working/imgEnhancement/Test'# update the input dir, which at least contains such sub-folder: low, global_score, local_prior
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    main(input_dir)
