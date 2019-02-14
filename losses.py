import torch
from torch import nn


# Use scale invariant loss for student learning from teacher
class ScaleInvariantLoss(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        predicted_depths, goal_depths, boundaries = x
        depth_ratio_map = torch.log(boundaries * predicted_depths + self.epsilon) - \
                          torch.log(boundaries * goal_depths + self.epsilon)

        weighted_sum = torch.sum(boundaries, dim=(1, 2, 3))
        loss_1 = torch.sum(depth_ratio_map * depth_ratio_map,
                           dim=(1, 2, 3)) / weighted_sum
        sum_2 = torch.sum(depth_ratio_map, dim=(1, 2, 3))
        loss_2 = (sum_2 * sum_2) / (weighted_sum * weighted_sum)
        return torch.mean(loss_1 + loss_2)


class NormalizedWeightedMaskedL2Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(NormalizedWeightedMaskedL2Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        depth_maps, warped_depth_maps, intersect_masks, translations = x
        # loss = torch.sum(torch.log(1.0 + torch.abs(intersect_masks * (depth_maps - warped_depth_maps))), dim=(1, 2, 3)) / (self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))

        translations = translations.view(-1, 3, 1)
        translation_norms = torch.sqrt(torch.sum(translations * translations, dim=(1, 2))).view(-1)
        translation_weights = (torch.tensor(1.0).float().cuda() / (torch.tensor(1.0e-8).float().cuda() + translation_norms)).view(-1)
        loss = torch.sum(intersect_masks * (depth_maps - warped_depth_maps) * (depth_maps - warped_depth_maps),
                         dim=(1, 2, 3), keepdim=False) / (0.5 * torch.sum(intersect_masks * (depth_maps * depth_maps + warped_depth_maps * warped_depth_maps), dim=(1, 2, 3), keepdim=False) + self.epsilon)
        loss = torch.sum(loss * translation_weights) / torch.sum(translation_weights)
        return loss


class SparseMaskedL1Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(SparseMaskedL1Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        flows, flows_from_depth, sparse_masks = x
        loss = torch.sum(sparse_masks * torch.abs(flows - flows_from_depth),
                         dim=(1, 2, 3)) / (self.epsilon + torch.sum(sparse_masks, dim=(1, 2, 3)))
        return torch.mean(loss)


class SparseMaskedL1LossDisplay(nn.Module):
    def __init__(self, epsilon=1.0):
        super(SparseMaskedL1LossDisplay, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        flows, flows_from_depth, sparse_masks = x
        loss = torch.sum(sparse_masks * torch.abs(flows - flows_from_depth),
                         dim=(1, 2, 3), keepdim=False) / (self.epsilon + torch.sum(sparse_masks, dim=(1, 2, 3), keepdim=False))
        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(MaskedL1Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        images, twice_warped_images, intersect_masks = x
        loss = torch.sum(intersect_masks * torch.abs(images - twice_warped_images), dim=(1, 2, 3)) / (
                self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))
        return torch.mean(loss)
