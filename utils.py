import torchvision
from torch.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np

import matplotlib.pyplot as plt
import tempfile
import IPython
from skimage.transform import rescale
from PIL import Image
from tqdm import tqdm

def pilimg_to_tensor(pil_img, device):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  t = t.to(device)
  return(t)

def display_as_pilimg(t, save = False, filename= None):
    t = t.to('cpu').squeeze()
    if t.min() < 0:
        t = (t + 1) / 2
    t = t - t.min() 
    t = t / t.max()
    t = t.clamp(0.,1.)
    pil_img = torchvision.transforms.ToPILImage()(t)
    if save: 
        pil_img.save(filename)
    display(pil_img)

def inpainting_operator(x, h=256, w=256, device = 'cuda:0'): 
    hcrop, wcrop = h//2, w//2
    corner_top, corner_left = h//4, int(0.45*w)
    mask = torch.ones(x.shape, device=device)
    mask[:,:,corner_top:corner_top+hcrop,corner_left:corner_left+wcrop] = 0
    return x*mask

def gaussian_kernel(kernel_size, sigma, device):
    kernel = torch.arange(0, kernel_size, device=device) - (kernel_size - 1) // 2
    x, y = torch.meshgrid(kernel, kernel, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def blurring_operator(x, kernel_size=5, sigma=10, device='cuda:0'):
    kt = gaussian_kernel(kernel_size, sigma, device)
    m, n = kernel_size, kernel_size
    b, c, M, N = x.shape
    k = torch.zeros((M,N),device=device)
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    k = k[None,None,:,:]
    fk = fft2(k)
    return ifft2(fft2(x)*fk).real



def transposed_blurring_op(x, kernel_size=5, sigma=2, device='cuda:0'):
    kt = gaussian_kernel(kernel_size, sigma, device)
    m, n = kernel_size, kernel_size
    b, c, M, N = x.shape
    k = torch.zeros((M,N),device=device)
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    k = k[None,None,:,:]
    fk = fft2(k)
    return ifft2(fft2(x)*torch.conj(fk)).real



def psnr(uref, ut,M=1):
    rmse = np.sqrt(np.mean((np.array(uref.cpu())-np.array(ut.cpu()))**2))
    return 20*np.log10(M/rmse)