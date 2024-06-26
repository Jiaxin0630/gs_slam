import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    depth_ratio = config["Training"]["depth_ratio"]
    gray_scale_threshold = config["Training"]["gray_scale_threshold"]
    gray_scale = 0.299*gt_image[0,...] + 0.587 * gt_image[1,...]  + 0.114 * gt_image[2,...]
    
    gray_mask = (gray_scale> gray_scale_threshold).view(*mask_shape) * (gray_scale < (1 - gray_scale_threshold)).view(*mask_shape)
    
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)
    
    l1_rgb = (opacity * torch.abs(image * gray_mask - gt_image * gray_mask)).mean()
    
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return (1-depth_ratio) * l1_rgb + depth_ratio * l1_depth.mean()


def get_loss_mapping(config, image, rendered_depth, viewpoint, opacity):
    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    m = (gt_depth > 0) * (rendered_depth > 0)
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - config['opt_params']['lambda_dssim']) * Ll1 + config['opt_params']['lambda_dssim'] * (1.0 - ssim(image, gt_image))
  
    loss = loss + \
            0.0*depth_ranking_loss((rendered_depth*m).squeeze(), (gt_depth*m).squeeze()) + \
            l1dep_loss((rendered_depth*m).squeeze(), (gt_depth*m).squeeze()) 
    
    return loss


def get_isotropic_loss_1(scaling, hp = 1.0):
    max,_ = torch.max(scaling,1)
    min,_ = torch.min(scaling,1)
    
    loss = 1.0/scaling.shape[0]*(torch.max(max/min,torch.tensor(hp))-hp).sum()
    return loss

def get_isotropic_loss_2(scaling):
    loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
    return loss

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1dep_loss(network_output, gt):
    m = (gt > 0) * (network_output > 0)
    # return torch.abs((network_output[m] - gt[m])).mean()
    return torch.abs((1/network_output[m] - 1/gt[m])).mean()

def depth_ranking_loss(rendered_depth, gt_depth):
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    """
    m = 1e-4
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    loss = torch.nanmean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))

    dpt_diff = gt_depth[:, ::2] - gt_depth[:, 1::2]
    out_diff = rendered_depth[:, ::2] - rendered_depth[:, 1::2] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    loss += torch.nanmean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
