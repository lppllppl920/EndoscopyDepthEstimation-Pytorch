import torch
import numpy as np
from torch import nn


class WeightedScaleInvariantLoss(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(WeightedScaleInvariantLoss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()

    def forward(self, x):
        absolute_depth_estimations, input_sparse_depths, input_sparse_masks = x

        depth_ratio_map = torch.where(input_sparse_depths < self.epsilon, self.zero,
                                      torch.log(absolute_depth_estimations + self.epsilon) -
                                      torch.log(input_sparse_depths))

        weighted_sum = torch.sum(input_sparse_masks, dim=(1, 2, 3))
        loss_1 = torch.sum(torch.mul(input_sparse_masks, depth_ratio_map * depth_ratio_map),
                           dim=(1, 2, 3)) / weighted_sum
        sum_2 = torch.sum(torch.mul(input_sparse_masks, depth_ratio_map), dim=(1, 2, 3))
        loss_2 = (sum_2 * sum_2) / (weighted_sum * weighted_sum)
        return torch.mean(loss_1 + loss_2)


class MaskedLogL1Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(MaskedLogL1Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        depth_maps, warped_depth_maps, intersect_masks = x
        # loss = torch.sum(torch.log(1.0 + intersect_masks * torch.abs(depth_maps - warped_depth_maps)),
        #                  dim=(1, 2, 3)) / (self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))

        loss = torch.sum(intersect_masks * torch.abs(depth_maps - warped_depth_maps),
                         dim=(1, 2, 3)) / (self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))
        return torch.mean(loss)


# TODO: Be sure to eliminate data contamination in the optical flows
class SparseMaskedL1Loss(nn.Module):
    def __init__(self):
        super(SparseMaskedL1Loss, self).__init__()

    def forward(self, x):
        opt_flows, opt_flows_from_depth, sparse_masks = x
        loss = torch.sum(sparse_masks * torch.abs(opt_flows - opt_flows_from_depth),
                         dim=(1, 2, 3)) / torch.sum(sparse_masks, dim=(1, 2, 3))
        return torch.mean(loss)


class MaskedL1Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(MaskedL1Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        images, twice_warped_images, intersect_masks = x
        loss = torch.sum(intersect_masks * torch.abs(images - twice_warped_images), dim=(1, 2, 3)) / (
                self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))
        return torch.mean(loss)
