from __future__ import division
import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, masks, intrinsics):
        for t in self.transforms:
            images, masks, intrinsics = t(images, masks, intrinsics)
        return images, masks, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, masks, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, masks, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, masks, intrinsics):
        tensors = []
        tensors_masks = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float() / 255)

        for mask in masks:
            mask = np.transpose(mask, (2, 0, 1))
            tensors_masks.append(torch.from_numpy(mask).float() / 255)

        return tensors, tensors_masks, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, masks, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            output_masks = [np.copy(np.fliplr(mask)) for mask in masks]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_masks = masks
            output_intrinsics = intrinsics
        return output_images, output_masks, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, masks, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [cv2.resize(im, dsize=(scaled_w, scaled_h)) for im in images]
        scaled_masks = [cv2.resize(mask, dsize=(scaled_w, scaled_h)).reshape((scaled_h, scaled_w, 1)) for mask in masks]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        cropped_masks = [mask[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for mask in scaled_masks]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, cropped_masks, output_intrinsics


class RandomColor:
    def __init__(self, prob=1.0, hue_limit=0.2, brightness_limit=0.2, saturation_limit=0.2):
        self.hue_limit = hue_limit
        self.brightness_limit = brightness_limit
        self.saturation_limit = saturation_limit
        self.prob = prob

    def __call__(self, images, masks, intrinsics):
        if random.random() < self.prob:
            transformed_images = []
            scale_brightness = np.random.uniform(low=-1.0, high=1.0)
            scale_saturation = np.random.uniform(low=-1.0, high=1.0)
            scale_hue = np.random.uniform(low=-1.0, high=1.0)
            for img in images:
                hsv = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV_FULL)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.brightness_limit * scale_brightness)
                hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
                hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + self.saturation_limit * scale_saturation)
                hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
                hsv[:, :, 0] = hsv[:, :, 0] * (1.0 + self.hue_limit * scale_hue)
                hsv[:, :, 0][hsv[:, :, 0] > 255] = 255
                img = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR_FULL)
                img = img.astype(np.float32)
                transformed_images.append(img)
            return transformed_images, masks, intrinsics
        else:
            return images, masks, intrinsics
