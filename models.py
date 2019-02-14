import torch
from torch import nn
from torch.nn import functional as F


# Removed dropout and changed the transition up layers in the original implementation
# to mitigate the grid patterns of the network output
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop_(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


def center_crop_(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=1):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        # Softmax

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out


def FCDenseNet57(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=6, wf=6, padding=True,
                 up_mode='upconv'):
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
                                                padding))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding))
            prev_channels = 2 ** (wf + i)
        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
                # x = F.max_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=int(padding),
                                         output_padding=int(padding))
            # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
            #                              stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))

        self.conv_block = UNetConvBlock(in_size, out_size, padding)

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


def images_warping(images, source_coord_w_flat, source_coord_h_flat, padding_mode="zeros"):
    batch_num, channels, image_h, image_w = images.shape
    warped_images_flat = _bilinear_interpolate(images.permute(0, 2, 3, 1), x=source_coord_w_flat,
                                               y=source_coord_h_flat, padding_mode=padding_mode)
    warped_images = warped_images_flat.view(batch_num, image_h, image_w, channels).permute(0, 3, 1, 2)
    return warped_images


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


class DepthScalingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthScalingLayer, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()
        self.one = torch.tensor(1.0).float().cuda()

    def forward(self, x):
        absolute_depth_estimations, input_sparse_depths, input_weighted_sparse_masks = x
        # Use sparse depth values which are greater than a certain ratio of the mean value of the sparse depths to avoid
        # unstability of scale recovery
        input_sparse_binary_masks = torch.where(input_weighted_sparse_masks > 1.0e-8, self.one, self.zero)
        mean_sparse_depths = torch.sum(input_sparse_depths * input_sparse_binary_masks, dim=(1, 2, 3), keepdim=True) / torch.sum(input_sparse_binary_masks, dim=(1, 2, 3), keepdim=True)
        above_mean_masks = torch.where(input_sparse_depths > 0.5 * mean_sparse_depths, self.one, self.zero)

        # Introduce a criteria to reduce the variation of scale maps
        sparse_scale_maps = input_sparse_depths * above_mean_masks / (self.epsilon + absolute_depth_estimations)
        mean_scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3), keepdim=True) / torch.sum(above_mean_masks, dim=(1, 2, 3), keepdim=True)
        centered_sparse_scale_maps = sparse_scale_maps - above_mean_masks * mean_scales
        scale_stds = torch.sqrt(torch.sum(centered_sparse_scale_maps * centered_sparse_scale_maps, dim=(1, 2, 3), keepdim=False) / torch.sum(above_mean_masks, dim=(1, 2, 3), keepdim=False))
        scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3)) / torch.sum(above_mean_masks, dim=(1, 2, 3))
        return torch.mul(scales.view(-1, 1, 1, 1), absolute_depth_estimations), torch.mean(scale_stds / mean_scales)


class FlowfromDepthLayer(torch.nn.Module):
    def __init__(self):
        super(FlowfromDepthLayer, self).__init__()

    def forward(self, x):
        depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        flow_image = _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                              intrinsic_matrices)
        return flow_image


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
def _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices):
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
    def __init__(self, epsilon=1.0e-8):
        super(DepthWarpingLayer, self).__init__()
        self.zero = torch.tensor(0.0).float().cuda()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        warped_depth_maps, intersect_masks = _depth_warping(depth_maps_1, depth_maps_2, img_masks,
                                                            translation_vectors,
                                                            rotation_matrices, intrinsic_matrices, self.epsilon)
        return warped_depth_maps, intersect_masks


# Warping depth map in coordinate system 2 to coordinate system 1
def _depth_warping(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                   intrinsic_matrices, epsilon):
    # Generate a meshgrid for each depth map to calculate value
    # BxHxWxC
    depth_maps_1 = torch.mul(depth_maps_1, img_masks)
    depth_maps_2 = torch.mul(depth_maps_2, img_masks)

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

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.where(img_masks > 0.5, depth_maps_2_calculate, epsilon)
    depth_maps_2_calculate = torch.where(depth_maps_2_calculate > 0.0, depth_maps_2_calculate, epsilon)

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

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
    intersect_masks = torch.where(_bilinear_interpolate(img_masks, u_2_flat, v_2_flat) * img_masks >= 0.9,
                               torch.tensor(1.0).float().cuda(),
                               torch.tensor(0.0).float().cuda()).view(num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]

