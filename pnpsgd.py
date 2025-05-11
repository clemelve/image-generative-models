import torch


def PnPSGD(x_true, y, nu, linear_operator, transp_op, denoiser, device, niter = 20): 

    tau = 1.9*nu**2
    s = 4*nu 
    x = y.clone()
    for it in range(niter):
        grad = transp_op(linear_operator(x, device=device) - y, device=device) /nu**2
        xn = x - tau * grad
        with torch.no_grad():
            xn = denoiser(xn, sigma=s)
        x = xn
    return x