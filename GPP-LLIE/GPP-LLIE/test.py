import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import glob
import os
import gc
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

def main(inp_dir):
    lr_dir = os.path.join(inp_dir, 'low')
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    local_prior_dir = os.path.join(inp_dir, 'local_prior')

    out_dir = os.path.join(inp_dir, 'outputs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    device = torch.device('cuda:0')
    
    # 1. Load to CPU to avoid the 'invalid load key' and VRAM spikes
    print("Loading weights...")
    state_dict = torch.load('/kaggle/working/BharathPreTrainedDS/weight_lol.pth', map_location='cpu')

    # 2. Initialize and load models
    model = DiT_incontext_revise().to(device)
    model.load_state_dict(state_dict['dit'])
    
    vae = AutoencoderKL().to(device)
    vae.load_state_dict(state_dict['vae'])
    
    cond_lq = CondEncoder().to(device)
    cond_lq.load_state_dict(state_dict['cond'])
    
    second_decoder = Decoder2().to(device)
    second_decoder.load_state_dict(state_dict['second_decoder'])

    # 3. CRITICAL: Free the CPU RAM immediately
    del state_dict 
    gc.collect() # Only extra addition to clear Python 3.8 memory
    torch.cuda.empty_cache()

    model.eval()
    vae.eval()
    cond_lq.eval()
    second_decoder.eval()

    diffusion_val = create_diffusion(str(25))
    to_tensor = ToTensor()

    for lr_path, global_path, local_path in zip(lr_paths, global_prior_paths, local_prior_paths):
        print(f"Processing: {os.path.basename(lr_path)}")
        
        # Ensure image is moved to GPU
        y = to_tensor(cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        global_prior = torch.load(global_path, map_location=device)
        local_prior = torch.load(local_path, map_location=device)

        with torch.no_grad():
            y_feat, enc_feat = cond_lq(y, True)
            b, c, h, w = y.shape
            z = torch.randn(1, 3, h // 4, w // 4, device=device)
            model_kwargs = dict(y=y_feat, vis=global_prior, q_map=local_prior)

            # Sampling images
            samples = diffusion_val.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=True, device=device
            )

            dec_feat = vae.decode(samples, mid_feat=True)
            sr = second_decoder(samples, dec_feat, enc_feat)
                  
        save_image(sr, os.path.join(out_dir, os.path.basename(lr_path)))
        
        # 4. Clear GPU cache after each image
        torch.cuda.empty_cache()

if __name__ == "__main__":
    input_dir = '/kaggle/working/imgEnhancement/Test'
    main(input_dir)