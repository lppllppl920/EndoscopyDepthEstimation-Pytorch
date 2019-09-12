import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, zeros_


def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
                                                                              list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = torch.bmm(intrinsics_inv, current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = torch.bmm(proj_c2p_rot, cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = torch.bmm(torch.bmm(xmat, ymat), zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert (intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = torch.bmm(intrinsics, pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
                                 padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    return projected_img


def resize_like(input, ref):
    assert (input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


def upconv(in_chnls, out_chnls):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chnls, out_chnls, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def downconv(in_chnls, out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size,
                  stride=2, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chnls, out_chnls, kernel_size,
                  stride=1, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


def conv(in_chnls, out_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def get_disparity(in_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


class DispNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01, training=True):
        super(DispNet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.training = training
        # encoder
        self.conv1 = downconv(3, 32, kernel_size=7)
        self.conv2 = downconv(32, 64, kernel_size=5)
        self.conv3 = downconv(64, 128, kernel_size=3)
        self.conv4 = downconv(128, 256, kernel_size=3)
        self.conv5 = downconv(256, 512, kernel_size=3)
        self.conv6 = downconv(512, 512, kernel_size=3)
        self.conv7 = downconv(512, 512, kernel_size=3)

        # decoder
        self.upconv7 = upconv(512, 512)
        self.upconv6 = upconv(512, 512)
        self.upconv5 = upconv(512, 256)
        self.upconv4 = upconv(256, 128)
        self.upconv3 = upconv(128, 64)
        self.upconv2 = upconv(64, 32)
        self.upconv1 = upconv(32, 16)

        self.iconv7 = conv(512 + 512, 512)
        self.iconv6 = conv(512 + 512, 512)
        self.iconv5 = conv(256 + 256, 256)
        self.iconv4 = conv(128 + 128, 128)
        self.iconv3 = conv(64 + 64 + 1, 64)
        self.iconv2 = conv(32 + 32 + 1, 32)
        self.iconv1 = conv(16 + 1, 16)

        self.disp4 = get_disparity(128)
        self.disp3 = get_disparity(64)
        self.disp2 = get_disparity(32)
        self.disp1 = get_disparity(16)

    def forward(self, x):
        # encode

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # decode
        out_upconv7 = resize_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = resize_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = resize_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = resize_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_disp4 = self.alpha * self.disp4(out_iconv4) + self.beta

        out_upconv3 = resize_like(self.upconv3(out_iconv4), out_conv2)
        out_updisp4 = resize_like(F.interpolate(
            out_disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, out_updisp4), 1)
        out_iconv3 = self.iconv3(concat3)
        out_disp3 = self.alpha * self.disp3(out_iconv3) + self.beta

        out_upconv2 = resize_like(self.upconv2(out_iconv3), out_conv1)
        out_updisp3 = resize_like(F.interpolate(
            out_disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, out_updisp3), 1)
        out_iconv2 = self.iconv2(concat2)
        out_disp2 = self.alpha * self.disp2(out_iconv2) + self.beta

        out_upconv1 = resize_like(self.upconv1(out_iconv2), x)
        out_updisp2 = resize_like(F.interpolate(
            out_disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, out_updisp2), 1)
        out_iconv1 = self.iconv1(concat1)
        out_disp1 = self.alpha * self.disp1(out_iconv1) + self.beta

        if self.training:
            return out_disp1, out_disp2, out_disp3, out_disp4
        else:
            return out_disp1


def downconv_pose(in_chnls, out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size=kernel_size,
                  stride=2, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


class PoseNet(nn.Module):

    def __init__(self, num_source):
        super(PoseNet, self).__init__()

        self.num_source = num_source

        self.conv1 = downconv_pose(3 * (1 + num_source), 16, 7)  # 1/2
        self.conv2 = downconv_pose(16, 32, 5)  # 1/4
        self.conv3 = downconv_pose(32, 64, 3)  # 1/8
        self.conv4 = downconv_pose(64, 128, 3)  # 1/16
        self.conv5 = downconv_pose(128, 256, 3)  # 1/32
        self.conv6 = downconv_pose(256, 256, 3)  # 1/64
        self.conv7 = downconv_pose(256, 256, 3)  # 1/128
        self.pred_poses = nn.Conv2d(256, 6 * self.num_source, kernel_size=1, padding=0)  # 1/128 shapes: bs,chnls,h,w

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_poses = self.pred_poses(out_conv7)
        out_avg_poses = torch.mean(out_poses, (2, 3))  # shapes: bs,6*num_src,h,w-> bs,6*num_src
        out_avg_poses = 0.01 * out_avg_poses.view(out_avg_poses.shape[0], self.num_source, 6)
        return out_avg_poses
