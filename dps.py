from math import e

import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from utils import pilimg_to_tensor, display_as_pilimg, psnr

class DPS:
  def __init__(self, model, timesteps, device):
    self.num_diffusion_timesteps = timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    beta_start = 0.0001
    beta_end = 0.02
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)
    self.device = device


  def get_eps_from_model(self, x, t):
    # the model outputs:
    # - an estimation of the noise eps (chanels 0 to 2)
    # - learnt variances for the posterior  (chanels 3 to 5)
    # (see Improved Denoising Diffusion Probabilistic Models
    # by Alex Nichol, Prafulla Dhariwal
    # for the parameterization)
    # We discard the second part of the output for this practice session.
    model_output = self.model(x, torch.tensor(t, device=self.device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sample(self, show_steps=True):
    with torch.no_grad():  # avoid backprop wrt model parameters
      x = torch.randn(self.imgshape,device=self.device)  # initialize x_t for t=T
      for i, t in enumerate(self.reversed_time_steps):
          z = torch.randn(self.imgshape,device=self.device) if i > 0 else 0
          eps = self.get_eps_from_model(x, t)
          x = np.sqrt(1.0 / self.alphas[t])*(x - self.betas[t]/np.sqrt(1-self.alphas_cumprod[t])*eps) + np.sqrt(self.betas[t])*z
          if i==0 or t%100==0 or t==0:
            print('Iteration:', i, '; Discrete time:', t)
            if show_steps: display_as_pilimg(x)
    return(x)

  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None):
    if vis_y is None:
        vis_y = y
    
    x = torch.randn(self.imgshape, device=self.device)

    for i, t in enumerate(self.reversed_time_steps):
        #print(t)
        with torch.no_grad():
            eps = self.get_eps_from_model(x, t)
            mu = torch.sqrt(torch.tensor(1.0 / self.alphas[t], device=self.device)) * (x - self.betas[t] / torch.sqrt(torch.tensor(1.0 - self.alphas_cumprod[t], device=self.device)) * eps)
        x = x.detach().requires_grad_(True)
        
        x_start = self.predict_xstart_from_eps(x, eps, t)
        loss = torch.linalg.vector_norm(linear_operator(x_start, device = self.device) - y)
        grad = torch.autograd.grad(loss**2, x, retain_graph=True)[0]

        with torch.no_grad():
            z = torch.randn(self.imgshape, device=self.device)
            zetaf = 0.1
            zeta = zetaf / loss.item()
            x = mu + torch.sqrt(torch.tensor(self.betas[t], device=self.device)) * z - zeta * grad

        """if i==0 or t%100==0 or t==0:
            print('Iteration:', i, '; Discrete time:', t)
            if show_steps:
                display_as_pilimg(x_start)
                if x_true is not None:
                    display_as_pilimg(x_true)
        """
    return x