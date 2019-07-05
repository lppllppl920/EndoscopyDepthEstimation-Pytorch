import torch
import utils
import torch
import torch.nn.functional as F

from . import models
from . import utils


# # TODO: We need to pay extra attention to the order of all these different indexes
# def flow_consistency_loss(multi_scale_batch_tgt_depths, batch_intrinsics, batch_intrinsics_inv, batch_multi_src_poses,
#                           batch_masks):
#     def one_scale(batch_tgt_depths, batch_masks, batch_intrinsics, batch_intrinsics_inv, batch_multi_src_poses):
#         b, _, h, w = batch_tgt_depths.shape
#         downscale = batch_tgt_depths.shape[2] / h
#
#         batch_masks_scaled = F.interpolate(batch_masks, (h, w), mode='nearest')
#         batch_intrinsics_scaled = torch.cat((batch_intrinsics[:, 0:2] / downscale, batch_intrinsics[:, 2:]), dim=1)
#         batch_intrinsics_scaled_inv = torch.cat(
#             (batch_intrinsics_inv[:, :, 0:2] * downscale, batch_intrinsics_inv[:, :, 2:]),
#             dim=2)
#
#         for i in range(batch_multi_src_poses.shape[1]):
#             batch_poses = batch_multi_src_poses[:, i].view(b, 6)
#             batch_pose_mats = utils.pose_vec2mat(batch_poses)
#             utils.compute_rigid_flow(batch_pose_mats, batch_tgt_depths)
#
#     loss = 0
#     for batch_tgt_depths in multi_scale_batch_tgt_depths:
#         loss += one_scale(batch_tgt_depths, batch_masks, batch_intrinsics, batch_intrinsics_inv, batch_multi_src_poses)
#     return loss


def depth_smoothness_loss(batch_tgt_imgs, multi_scale_batch_depths, batch_masks):
    def one_scale(batch_tgt_imgs, batch_depths, batch_masks):
        batch_size, _, height, width = batch_depths.shape
        batch_tgt_imgs_scaled = F.interpolate(batch_tgt_imgs, size=(height, width), mode='bilinear')
        batch_masks_scaled = F.interpolate(batch_masks, size=(height, width), mode='nearest')

        gradient_depth_x = gradient_x(batch_depths)  # shape: bs,1,h,w
        gradient_depth_y = gradient_y(batch_depths)

        gradient_img_x = gradient_x(batch_tgt_imgs_scaled)  # shape: bs,3,h,w
        gradient_img_y = gradient_y(batch_tgt_imgs_scaled)

        exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x), 1, True))  # shape: bs,1,h,w
        exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y), 1, True))

        smooth_x = gradient_depth_x * exp_gradient_img_x
        smooth_y = gradient_depth_y * exp_gradient_img_y

        loss = torch.sum(batch_masks_scaled * (torch.abs(smooth_x) + torch.abs(smooth_y)), dim=(1, 2, 3)) / torch.sum(
            batch_masks_scaled, dim=(1, 2, 3))
        return torch.mean(loss)

    if type(multi_scale_batch_depths) not in [list, tuple]:
        multi_scale_batch_depths = [multi_scale_batch_depths]

    loss = 0
    for d in multi_scale_batch_depths:
        loss += one_scale(batch_tgt_imgs, d, batch_masks)
    return loss


def photometric_reconstruction_loss(batch_tgt_imgs, batch_multi_src_ref_imgs, batch_masks, batch_intrinsics,
                                    batch_intrinsics_inv, multi_scale_batch_depths,
                                    batch_multi_src_pose, alpha, rotation_mode='euler', padding_mode='zeros'):
    def one_scale(batch_depths, batch_masks):
        reconstruction_loss = 0
        b, num_src, _, h, w = batch_multi_src_ref_imgs.shape
        downscale = batch_tgt_imgs.size(2) / h

        batch_tgt_imgs_scaled = F.interpolate(batch_tgt_imgs, (h, w), mode='bilinear')
        batch_masks_scaled = F.interpolate(batch_masks, (h, w), mode='nearest')
        # ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='bilinear') for ref_img in batch_multi_src_ref_imgs]

        batch_multi_src_ref_imgs_scaled = F.interpolate(batch_multi_src_ref_imgs.view(b, num_src * 3, h, w), (h, w),
                                                        mode='bilinear').view(b, num_src, 3, h, w)

        batch_intrinsics_scaled = torch.cat((batch_intrinsics[:, 0:2] / downscale, batch_intrinsics[:, 2:]), dim=1)
        batch_intrinsics_scaled_inv = torch.cat(
            (batch_intrinsics_inv[:, :, 0:2] * downscale, batch_intrinsics_inv[:, :, 2:]),
            dim=2)

        for i in range(batch_multi_src_ref_imgs_scaled.shape[1]):
            batch_ref_imgs = batch_multi_src_ref_imgs_scaled[:, i].view(b, 3, h, w)
            current_pose = batch_multi_src_pose[:, i].view(b, 6)
            batch_ref_imgs_warped = models.inverse_warp(batch_ref_imgs, batch_depths.squeeze(dim=1), current_pose,
                                                        batch_intrinsics_scaled,
                                                        batch_intrinsics_scaled_inv,
                                                        rotation_mode, padding_mode)
            out_of_bound = 1 - (batch_ref_imgs_warped == 0).prod(1, keepdim=True).type_as(batch_ref_imgs_warped)
            # diff = (tgt_img_scaled - ref_img_warped) * out_of_bound * mask_scaled
            diff = image_similarity(alpha=alpha, x=batch_tgt_imgs_scaled,
                                    y=batch_ref_imgs_warped) * out_of_bound * batch_masks_scaled
            reconstruction_loss += torch.mean(
                torch.sum(diff, dim=(1, 2, 3)) / torch.sum(out_of_bound * batch_masks_scaled, dim=(1, 2, 3)))

        return reconstruction_loss

    if type(batch_multi_src_ref_imgs) not in [list, tuple]:
        batch_multi_src_ref_imgs = [batch_multi_src_ref_imgs]

    loss = 0
    for d in multi_scale_batch_depths:
        loss += one_scale(d, batch_masks)
    return loss


def DSSIM(x, y):
    ''' Official implementation
    def SSIM(self, x, y):
        C1 = 0.01 ** 2 # why not use L=255
        C2 = 0.03 ** 2 # why not use L=255

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        # if this implementatin equvalent to the SSIM paper?
        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    '''
    # TODO: padding depend on the size of the input image sequences
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    sigma_x = avepooling2d(x ** 2) - mu_x ** 2
    sigma_y = avepooling2d(y ** 2) - mu_y ** 2
    sigma_xy = avepooling2d(x * y) - mu_x * mu_y
    k1_square = 0.01 ** 2
    k2_square = 0.03 ** 2
    L_square = 1
    SSIM_n = (2 * mu_x * mu_y + k1_square * L_square) * (2 * sigma_xy + k2_square * L_square)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + k1_square * L_square) * \
             (sigma_x + sigma_y + k2_square * L_square)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)


def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def image_similarity(alpha, x, y):
    return alpha * DSSIM(x, y) + (1 - alpha) * torch.abs(x - y)


def flow_smooth_loss(flow, img):
    smoothness = 0
    for i in range(2):
        smoothness += smooth_loss(flow[:, i, :, :].unsqueeze(1), img)
    return smoothness / 2

# TODO: geometric consistency loss?
