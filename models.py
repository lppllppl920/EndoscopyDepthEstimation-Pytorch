import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=6, wf=6, padding=True,
                 batch_norm=True, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                # x = F.avg_pool2d(x, 2)
                x = F.max_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=int(padding),
                                         output_padding=int(padding))
            # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
            #                              stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]


def _bilinear_interpolate(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    # Range [-1, 1]
    grid = torch.cat([torch.tensor(2.0).float().cuda() *
                      (x.view(num_batch, height, width, 1) / torch.tensor(width).float().cuda())
                      - torch.tensor(1.0).float().cuda(), torch.tensor(2.0).float().cuda() * (
                              y.view(num_batch, height, width, 1) / torch.tensor(width).float().cuda()) - torch.tensor(
        1.0).float().cuda()], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode).permute(0, 2, 3, 1)


def images_warping(images, source_coord_w_flat, source_coord_h_flat, padding_mode="zeros"):
    batch_num, channels, image_h, image_w = images.shape
    warped_images_flat = _bilinear_interpolate(images.permute(0, 2, 3, 1), x=source_coord_w_flat,
                                               y=source_coord_h_flat, padding_mode=padding_mode)
    warped_images = warped_images_flat.view(batch_num, image_h, image_w, channels).permute(0, 3, 1, 2)
    return warped_images


def images_rotation_coordinates_calculate(thetas, image_h, image_w):
    # B x 1 x 1
    cos_theta = torch.cos(thetas).view(-1, 1, 1)
    sin_theta = torch.sin(thetas).view(-1, 1, 1)

    image_center_h = torch.tensor(np.floor(image_h / 2.0)).float().cuda()
    image_center_w = torch.tensor(np.floor(image_w / 2.0)).float().cuda()

    h_grid, w_grid = torch.meshgrid(
        [torch.arange(start=0, end=image_h, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=image_w, dtype=torch.float32).cuda()])

    # 1 x H x W
    h_grid = h_grid.view(1, image_h, image_w)
    w_grid = w_grid.view(1, image_h, image_w)

    # B x H x W
    source_coord_h = cos_theta * (h_grid - image_center_h) + \
                     sin_theta * (w_grid - image_center_w) + image_center_h
    source_coord_w = -sin_theta * (h_grid - image_center_h) + \
                     cos_theta * (w_grid - image_center_w) + image_center_w

    source_coord_h_flat = source_coord_h.view(-1)
    source_coord_w_flat = source_coord_w.view(-1)

    return source_coord_h_flat, source_coord_w_flat


# TODO: Should we sum all elements together and then divide or first divide and
#  then sum them up to get the global scale estimation?
class DepthScalingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthScalingLayer, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()
        self.one = torch.tensor(1.0).float().cuda()

    def forward(self, x):
        absolute_depth_estimations, input_sparse_depths, input_sparse_masks = x
        input_sparse_masks = torch.where(input_sparse_masks < self.epsilon, self.zero, self.one)
        scales = torch.sum(input_sparse_depths, dim=(1, 2, 3)) / (
                self.epsilon + torch.sum(input_sparse_masks * absolute_depth_estimations, dim=(1, 2, 3)))
        # scales = torch.sum(torch.where(input_sparse_masks < 0.5, self.zero, input_sparse_depths / (self.epsilon +
        #                                                                                            absolute_depth_estimations)),
        #                    dim=(1, 2, 3)) / torch.sum(input_sparse_masks, dim=(1, 2, 3))
        return torch.mul(scales.view(-1, 1, 1, 1), absolute_depth_estimations)


class OpticalFlowConsistencyLayer(torch.nn.Module):
    def __init__(self):
        super(OpticalFlowConsistencyLayer, self).__init__()

    def forward(self, x):
        colors_2, depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, \
        translation_vectors_inverse, rotation_matrices_inverse, intrinsic_matrices = x
        intersect_masks, twice_warped_colors_2 = _2D_correspondence_consistency(colors_2, depth_maps_1, depth_maps_2,
                                                                                img_masks,
                                                                                translation_vectors, rotation_matrices,
                                                                                translation_vectors_inverse,
                                                                                rotation_matrices_inverse,
                                                                                intrinsic_matrices)
        return [intersect_masks, twice_warped_colors_2]

        # depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, \
        # translation_vectors_inverse, rotation_matrices_inverse, intrinsic_matrices = x
        #
        # intersect_masks, twice_warped_xy_grid, xy_grid = _opt_flow_consistency(depth_maps_1, depth_maps_2, img_masks,
        #                                                               translation_vectors, rotation_matrices,
        #                                                               translation_vectors_inverse,
        #                                                               rotation_matrices_inverse, intrinsic_matrices)
        # return [intersect_masks, twice_warped_xy_grid, xy_grid]


# # Optical flow for frame 1 to frame 2
# def _opt_flow_consistency(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
#                           translation_vectors_inverse, rotation_matrices_inverse,
#                           intrinsic_matrices):
#     zero = torch.tensor(0.0).float().cuda()
#     one = torch.tensor(1.0).float().cuda()
#
#     # Generate a meshgrid for each depth map to calculate value
#     # BxHxWxC
#     depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
#     depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)
#     img_masks = img_masks.permute(0, 2, 3, 1)
#     img_masks = torch.where(img_masks >= 0.95, one, zero)
#
#     num_batch, height, width, channels = depth_maps_1.shape
#
#     y_grid, x_grid = torch.meshgrid(
#         [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
#          torch.arange(start=0, end=width, dtype=torch.float32).cuda()])
#
#     x_grid = (x_grid / torch.tensor(width).float().cuda()).view(1, height, width, 1)
#     y_grid = (y_grid / torch.tensor(height).float().cuda()).view(1, height, width, 1)
#
#     u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
#                                          intrinsic_matrices)
#     u_2_flat = u_2.view(-1)
#     v_2_flat = v_2.view(-1)
#
#     xy_grid = torch.cat([x_grid, y_grid], dim=-1).expand(num_batch, -1, -1, -1)
#     warped_xy_grid = _bilinear_interpolate(xy_grid, u_2_flat, v_2_flat, padding_mode="border").view(num_batch, height,
#                                                                                                     width, 2)
#     warped_masks = _bilinear_interpolate(img_masks, u_2_flat, v_2_flat, padding_mode="zeros").view(num_batch, height,
#                                                                                                    width, 1)
#
#     # Let's warp the warped_masks and warped_xy_grid again back to coordinate 2
#     u_1, v_1 = _warp_coordinate_generate(depth_maps_2, img_masks, translation_vectors_inverse,
#                                          rotation_matrices_inverse,
#                                          intrinsic_matrices)
#
#     u_1_flat = u_1.view(-1)
#     v_1_flat = v_1.view(-1)
#
#     twice_warped_xy_grid = _bilinear_interpolate(warped_xy_grid, u_1_flat, v_1_flat, padding_mode="border").view(
#         num_batch, height, width, 2)
#     twice_warped_masks = _bilinear_interpolate(warped_masks, u_1_flat, v_1_flat, padding_mode="zeros").view(num_batch,
#                                                                                                             height,
#                                                                                                             width, 1)
#     last_warped_masks = _bilinear_interpolate(img_masks, u_1_flat, v_1_flat, padding_mode="zeros").view(num_batch,
#                                                                                                         height, width,
#                                                                                                         1)
#     #
#     # twice_warped_masks = torch.where(_nearest_interpolate(warped_masks, u_1_flat, v_1_flat) >= 0.95,
#     #                                  one, zero).view(num_batch, height, width, 1)
#
#     # last_warped_masks = torch.where(_nearest_interpolate(img_masks, u_1_flat, v_1_flat) >= 0.95,
#     #                                 one, zero).view(num_batch, height, width, 1)
#
#     return [torch.where((img_masks * last_warped_masks * twice_warped_masks) >= 0.95, one, zero).permute(0, 3, 1, 2),
#             twice_warped_xy_grid.permute(0, 3, 1, 2), xy_grid.permute(0, 3, 1, 2)]

# Optical flow for frame 1 to frame 2
def _2D_correspondence_consistency(colors_2, depth_maps_1, depth_maps_2, img_masks, translation_vectors,
                                   rotation_matrices,
                                   translation_vectors_inverse, rotation_matrices_inverse,
                                   intrinsic_matrices):
    zero = torch.tensor(0.0).float().cuda()
    one = torch.tensor(1.0).float().cuda()

    # Generate a meshgrid for each depth map to calculate value
    # BxHxWxC
    colors_2 = colors_2.permute(0, 2, 3, 1)
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)
    img_masks = torch.where(img_masks >= 0.9, one, zero)

    num_batch, height, width, channels = colors_2.shape

    # y_grid, x_grid = torch.meshgrid(
    #     [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
    #      torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    # x_grid = (x_grid / torch.tensor(width).float().cuda()).view(1, height, width, 1)
    # y_grid = (y_grid / torch.tensor(height).float().cuda()).view(1, height, width, 1)

    u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                         intrinsic_matrices)
    u_2_flat = u_2.view(-1)
    v_2_flat = v_2.view(-1)

    # xy_grid = torch.cat([x_grid, y_grid], dim=-1).expand(num_batch, -1, -1, -1)
    warped_colors = _bilinear_interpolate(colors_2, u_2_flat, v_2_flat, padding_mode="border").view(num_batch, height,
                                                                                                    width, channels)
    warped_masks = _bilinear_interpolate(img_masks, u_2_flat, v_2_flat, padding_mode="zeros").view(num_batch, height,
                                                                                                   width, 1)

    # Let's warp the warped_masks and warped_xy_grid again back to coordinate 2
    u_1, v_1 = _warp_coordinate_generate(depth_maps_2, img_masks, translation_vectors_inverse,
                                         rotation_matrices_inverse,
                                         intrinsic_matrices)

    u_1_flat = u_1.view(-1)
    v_1_flat = v_1.view(-1)
    twice_warped_colors = _bilinear_interpolate(warped_colors, u_1_flat, v_1_flat, padding_mode="border").view(
        num_batch, height, width, channels)
    twice_warped_masks = _bilinear_interpolate(warped_masks, u_1_flat, v_1_flat, padding_mode="zeros").view(num_batch,
                                                                                                            height,
                                                                                                            width, 1)

    last_warped_masks = _bilinear_interpolate(img_masks, u_1_flat, v_1_flat, padding_mode="zeros").view(num_batch,
                                                                                                        height, width,
                                                                                                        1)

    return [torch.where((img_masks * last_warped_masks * twice_warped_masks) >= 0.9, one, zero).permute(0, 3, 1, 2),
            twice_warped_colors.permute(0, 3, 1, 2)]


class OpticalFlowfromDepthLayer(torch.nn.Module):
    def __init__(self):
        super(OpticalFlowfromDepthLayer, self).__init__()

    def forward(self, x):
        depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        opt_flow_image = _opt_flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                              intrinsic_matrices)
        return opt_flow_image


def _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices):
    # Generate a meshgrid for each depth map to calculate value
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda()

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda().view(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.gesv(eye, intrinsic_matrices)

    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # depth_maps_1 = depth_maps_1 * img_masks

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.tensor(1.0e30).float().cuda() * (torch.tensor(1.0).float().cuda() - img_masks) + \
                             img_masks * depth_maps_2_calculate

    # torch.where(img_masks > 0.0, depth_maps_2_calculate, torch.from_numpy(np.array([1.0e30])).float().cuda())

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate
    return [u_2, v_2]


# Optical flow for frame 1 to frame 2
def _opt_flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices):
    # BxHxWxC
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                         intrinsic_matrices)

    return torch.cat(
        [(u_2 - x_grid) / torch.tensor(width).float().cuda(), (v_2 - y_grid) / torch.tensor(height).float().cuda()],
        dim=-1).permute(0, 3, 1, 2)


class DepthWarpingLayer(torch.nn.Module):
    def __init__(self):
        super(DepthWarpingLayer, self).__init__()
        self.zero = torch.tensor(0.0).float().cuda()

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        warped_depth_maps, intersect_masks = _depth_warping(depth_maps_1, depth_maps_2, img_masks,
                                                            translation_vectors,
                                                            rotation_matrices, intrinsic_matrices)
        return warped_depth_maps, intersect_masks


# Warping depth map in coordinate system 2 to coordinate system 1
def _depth_warping(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                   intrinsic_matrices):
    # Generate a meshgrid for each depth map to calculate value
    # BxHxWxC
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)

    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda()

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda().view(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.gesv(eye, intrinsic_matrices)

    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    # The following is when we have same intrinsic matrix assumption
    # temp_mat = torch.matmul(intrinsic_matrix, rotation_matrices_inverse)
    # W = torch.bmm(temp_mat, -translation_vectors)
    # M = torch.matmul(temp_mat, intrinsic_matrix_inverse)
    #
    # W_2 = torch.matmul(intrinsic_matrix, translation_vectors)
    # M_2 = torch.matmul(torch.matmul(intrinsic_matrix, rotation_matrices), intrinsic_matrix_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.where(img_masks > 1.0e-3, depth_maps_2_calculate,
                                         torch.from_numpy(np.array([1.0e30])).float().cuda())

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate

    W_2 = torch.bmm(intrinsic_matrices, translation_vectors)
    M_2 = torch.bmm(torch.bmm(intrinsic_matrices, rotation_matrices), intrinsic_matrices_inverse)

    temp = torch.matmul(M_2.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3).narrow(dim=-1, start=2,
                                                                                               length=1).view(-1,
                                                                                                              height,
                                                                                                              width, 1)
    depth_maps_1_calculate = W_2.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_2, temp)
    depth_maps_1_calculate = torch.mul(img_masks, depth_maps_1_calculate)

    u_2_flat = u_2.view(-1)
    v_2_flat = v_2.view(-1)

    warped_depth_maps_2 = _bilinear_interpolate(depth_maps_1_calculate, u_2_flat, v_2_flat).view(num_batch, 1, height,
                                                                                                 width)

    # binarize
    warped_masks = torch.where(_bilinear_interpolate(img_masks, u_2_flat, v_2_flat) >= 0.9,
                               torch.tensor(1.0).float().cuda(),
                               torch.tensor(0.0).float().cuda()).view(num_batch, height, width, 1)
    intersect_masks = (warped_masks * img_masks).view(num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]
