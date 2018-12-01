"""
Based on a set of transformations developed by Alexander Buslaev as a part of the winning solution (1 out of 735)
to the Kaggle: Carvana Image Masking Challenge.

https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/transforms.py
"""

import random
import cv2
import numpy as np
import math


class MaskErodeDilation:
    def __init__(self, kernel_size_lower=0, kernel_size_upper=6):
        self.kernel_size_lower = kernel_size_lower
        self.kernel_size_upper = kernel_size_upper
        self.prob = 0.5

    def __call__(self, mask):
        kernel_size = random.randint(self.kernel_size_lower, self.kernel_size_upper)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if kernel_size == 0:
            return mask

        if random.random() < self.prob:
            return cv2.erode(mask, kernel, iterations=1)
        else:
            return cv2.dilate(mask, kernel, iterations=1)


class MaskShift:
    def __init__(self, limit=30, prob=1.0):
        self.limit = limit
        self.prob = prob

    def __call__(self, mask):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width = mask.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            msk1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_CONSTANT, value=-1.0)
            mask = msk1[int(y1):int(y2), int(x1):int(x2)]

        return mask


class MaskShiftScaleRotate:
    def __init__(self, shift_limit=0.5, scale_lower=0.5, scale_upper=1.5, rotate_limit=10, prob=1.0):
        self.shift_limit = shift_limit
        self.scale_lower = scale_lower
        self.scale_upper = scale_upper
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, mask):
        if random.random() < self.prob:
            height, width = mask.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(self.scale_lower, self.scale_upper)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            mask = cv2.warpPerspective(mask, mat, (width, height),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        return mask


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask


class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        for t in self.trans:
            x = t(x)
        return x, mask

class ImageRealOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x

class MaskOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask):
        for t in self.trans:
            mask = t(mask)
        return x, mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask

class RandomColor:
    def __init__(self, prob=1.0, hue_limit=0.05, brightness_limit=0.2, saturation_limit=0.2, contrast_limit=0.2):
        self.hue_limit = hue_limit
        self.brightness_limit = brightness_limit
        self.saturation_limit = saturation_limit
        self.contrast_limit = contrast_limit
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.brightness_limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values

            hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + self.saturation_limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255 #reset out of range values

            hsv[:, :, 0] = hsv[:, :, 0] * (1.0 + self.hue_limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 0][hsv[:, :, 0] > 179] = 179  # reset out of range values

            img = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

            alpha = 1.0 + self.contrast_limit * random.uniform(-1, 1)
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)

        return img

class Resize:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img, mask=None):
        img = cv2.resize(img, dsize=(self.w, self.h))
        if mask is not None:
            mask = cv2.resize(mask, dsize=(self.w, self.h))
        return img, mask

class ResizeImage:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        img = cv2.resize(img, dsize=(self.w, self.h))
        return img


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w]

        return img, mask


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2, :]

        return img, mask


class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
                   else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
                mask = (msk1[y1:y2, x1:x2, :] if size == size0
                        else cv2.resize(msk1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class CenterCrop:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2]

        if mask is not None:
            mask = mask[y1:y2, x1:x2]

        return img, mask


class ColorizationNormalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gray):
        max_pixel_value = 255.0

        img = img.astype(np.float32) / max_pixel_value
        gray = gray.astype(np.float32) / max_pixel_value

        gray -= np.ones(gray.shape) * self.mean
        gray /= np.ones(gray.shape) * self.std

        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std

        return img, gray


class Normalize:
    def __init__(self, normalize_mask=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.normalize_mask = normalize_mask

    def __call__(self, img, mask=None):
        max_pixel_value = 255.0

        img = img.astype(np.float32) / max_pixel_value

        if(mask != None):
            mask = mask.astype(np.float32)
            if (self.normalize_mask == True):
                mask /= 255.0

            mask -= np.ones(mask.shape) * 0.5
            mask /= np.ones(mask.shape) * 0.5

        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        return img, mask

class NormalizeImage:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.max_val = 255.0
    def __call__(self, img):
        img = img.astype(np.float32) / self.max_val
        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        return img

class Distort1:
    """"
    ## unconverntional augmnet ################################################################################3
    ## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

    ## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    ## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
    ## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

    ## barrel\pincushion distortion
    """

    def __init__(self, distort_limit=0.35, shift_limit=0.25, prob=0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            if 0:
                img = img.copy()
                for x in range(0, width, 10):
                    cv2.line(img, (x, 0), (x, height), (1, 1, 1), 1)
                for y in range(0, height, 10):
                    cv2.line(img, (0, y), (width, y), (1, 1, 1), 1)

            k = random.uniform(-self.distort_limit, self.distort_limit) * 0.00001
            dx = random.uniform(-self.shift_limit, self.shift_limit) * width
            dy = random.uniform(-self.shift_limit, self.shift_limit) * height

            #  map_x, map_y =
            # cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
            # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
            # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
            x, y = np.mgrid[0:width:1, 0:height:1]
            x = x.astype(np.float32) - width / 2 - dx
            y = y.astype(np.float32) - height / 2 - dy
            theta = np.arctan2(y, x)
            d = (x * x + y * y) ** 0.5
            r = d * (1 + k * d * d)
            map_x = r * np.cos(theta) + width / 2 + dx
            map_y = r * np.sin(theta) + height / 2 + dy

            img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return img, mask


class Distort2:
    """
    #http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    ## grid distortion
    """

    def __init__(self, num_steps=10, distort_limit=0.2, prob=0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            x_step = width // self.num_steps
            xx = np.zeros(width, np.float32)
            prev = 0
            for x in range(0, width, x_step):
                start = x
                end = x + x_step
                if end > width:
                    end = width
                    cur = width
                else:
                    cur = prev + x_step * (1 + random.uniform(-self.distort_limit, self.distort_limit))

                xx[start:end] = np.linspace(prev, cur, end - start)
                prev = cur

            y_step = height // self.num_steps
            yy = np.zeros(height, np.float32)
            prev = 0
            for y in range(0, height, y_step):
                start = y
                end = y + y_step
                if end > width:
                    end = height
                    cur = height
                else:
                    cur = prev + y_step * (1 + random.uniform(-self.distort_limit, self.distort_limit))

                yy[start:end] = np.linspace(prev, cur, end - start)
                prev = cur

            map_x, map_y = np.meshgrid(xx, yy)
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)
            img = cv2.remap(img, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.remap(mask, map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class RandomFilter:
    """
    blur sharpen, etc
    """

    def __init__(self, limit=.5, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32) / 9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1 - alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img


# https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
# color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

class RandomColorDual:
    def __init__(self, limit=0.1, prob=0.8):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        temp = mask.copy()
        if random.random() < self.prob:
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values

            hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + self.limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255 #reset out of range values
            mask = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(mask[..., :3])
            dtype = mask.dtype
            mask[:, :, :3] = clip(alpha * mask[:, :, :3] + gray, dtype, maxval)

            img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)

        return img, temp

class RandomBrightnessDual:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
            mask = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)

        return img, mask

class RandomContrastDual:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(mask[..., :3])
            dtype = mask.dtype
            mask[:, :, :3] = clip(alpha * mask[:, :, :3] + gray, dtype, maxval)
            img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)

        return img, mask

class RandomSaturationDual:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        # dont work :(
        if random.random() < self.prob:
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + self.limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255  # reset out of range values
            mask = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)
        return img, mask

class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.limit * np.random.uniform(low=-1.0, high=1.0))
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
            img = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        return img


class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        # dont work :(
        if random.random() < self.prob:
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
            img[..., :3] = clip(img[..., :3], dtype, maxval)
        return img


class RandomHueSaturationValue:
    def __init__(self, hue_shift_limit=(-10, 10), sat_shift_limit=(-25, 25), val_shift_limit=(-25, 25), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


def augment(x, mask=None, prob=0.5):
    return DualCompose([
        OneOrOther(
            *(OneOf([
                Distort1(distort_limit=0.05, shift_limit=0.05),
                Distort2(num_steps=2, distort_limit=0.05)]),
              ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45)), prob=prob),
        RandomFlip(prob=0.5),
        Transpose(prob=0.5),
        ImageOnly(RandomContrast(limit=0.2, prob=0.5)),
        ImageOnly(RandomFilter(limit=0.5, prob=0.2)),
    ])(x, mask)


class RandomNoise:
    def __init__(self, p=1.0, bias=True):
        self.prob = p
        self.bias = bias

    def __call__(self, image):
        if random.random() < self.prob:
            prob = random.random()
            if prob >= 0.0 and prob < 0.33:
                image = noisy("gauss", image)
            elif prob >= 0.33 and prob < 0.67:
                image = noisy("s&p", image)
            else:
                image = noisy("poisson", image)
            return image
        else:
            return image


def noisy(noise_typ, image, bias=True):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        if bias:
            image_bias = image * 0.5 + 0.5
            vals = len(np.unique(image_bias))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = image + np.random.poisson(image_bias * vals) / float(vals)
            # noisy = image + 1.0e-6 * np.random.poisson(image_bias)
        else:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = image + np.random.poisson(image * vals) / float(vals)
            # noisy = image + 1.0e-6  * np.random.poisson(image)
        return noisy
    # elif noise_typ == "speckle":
    #     row, col, ch = image.shape
    #     gauss = np.random.randn(row, col, ch)
    #     gauss = gauss.reshape(row, col, ch)
    #     noisy = image + image * gauss
    #     return noisy

    return image
