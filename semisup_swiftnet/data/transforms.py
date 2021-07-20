import numpy as np
import random
import pickle
import torch
from PIL import Image as pimg

from .util import *

RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Open:
    def __init__(self, palette=None, copy_labels=True):
        self.palette = palette
        self.copy_labels = copy_labels

    def __call__(self, example: dict):
        try:
            ret_dict = {}
            for k in ['image', 'image_next', 'image_prev']:
                if k in example:
                    ret_dict[k] = pimg.open(example[k]).convert('RGB')
                    if k == 'image':
                        ret_dict['target_size'] = ret_dict['image'].size
            if 'depth' in example:
                example['depth'] = pimg.open(example['depth'])
            if 'labels' in example:
                ret_dict['labels'] = pimg.open(example['labels'])
                if self.palette is not None:
                    ret_dict['labels'].putpalette(self.palette)
                if self.copy_labels:
                    ret_dict['original_labels'] = ret_dict['labels'].copy()
            if 'flow' in example:
                ret_dict['flow'] = readFlow(example['flow'])
        except OSError:
            print(example)
            raise
        return {**example, **ret_dict}


class SetTargetSize:
    def __init__(self, target_size, target_size_feats, stride=4):
        self.target_size = target_size
        self.target_size_feats = target_size_feats
        self.stride = stride

    def __call__(self, example):
        if all([self.target_size, self.target_size_feats]):
            example['target_size'] = self.target_size[::-1]
            example['target_size_feats'] = self.target_size_feats[::-1]
        else:
            k = 'original_labels' if 'original_labels' in example else 'image'
            example['target_size'] = example[k].shape[-2:]
            example['target_size_feats'] = tuple([s // self.stride for s in example[k].shape[-2:]])
        example['alphas'] = [-1]
        example['target_level'] = 0
        return example


class Tensor:
    def _trans(self, img, dtype):
        img = np.array(img, dtype=dtype)
        if len(img.shape) == 3:
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(img)

    def __call__(self, example):
        ret_dict = {}
        for k in ['image', 'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k], np.float32)
        if 'depth' in example:
            ret_dict['depth'] = self._trans(example['depth'], np.uint8)
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], np.int64)
        if 'original_labels' in example:
            ret_dict['original_labels'] = self._trans(example['original_labels'], np.int64)
        if 'depth_hist' in example:
            ret_dict['depth_hist'] = [self._trans(d, np.float32) for d in example['depth_hist']] if isinstance(
                example['depth_hist'], list) else self._trans(example['depth_hist'], np.float32)
        if 'pyramid' in example:
            ret_dict['pyramid'] = [self._trans(p, np.float32) for p in example['pyramid']]
        if 'pyramid_ms' in example:
            ret_dict['pyramid_ms'] = [[self._trans(p, np.float32) for p in pyramids] for pyramids in
                                      example['pyramid_ms']]
        if 'mux_indices' in example:
            ret_dict['mux_indices'] = torch.stack([torch.from_numpy(midx.flatten()) for midx in example['mux_indices']])
        if 'mux_masks' in example:
            ret_dict['mux_masks'] = [torch.from_numpy(np.uint8(mi)).unsqueeze(0) for mi in example['mux_masks']]
        if 'depth_bins' in example:
            ret_dict['depth_bins'] = torch.stack([torch.from_numpy(b) for b in example['depth_bins']])
        if 'flow' in example:
            # ret_dict['flow'] = torch.from_numpy(example['flow']).permute(2, 0, 1).contiguous()
            ret_dict['flow'] = torch.from_numpy(np.ascontiguousarray(example['flow']))
        # if 'flow_next' in example:
        #     ret_dict['flow_next'] = torch.from_numpy(example['flow_next']).permute(2, 0, 1 ).contiguous()
        if 'flow_sub' in example:
            # ret_dict['flow_sub'] = torch.from_numpy(example['flow_sub']).permute(2, 0, 1).contiguous()
            ret_dict['flow_sub'] = torch.from_numpy(np.ascontiguousarray(example['flow_sub']))
        if 'flipped' in example:
            del example['flipped']
        return {**example, **ret_dict}


class RandomSquareCropAndScale:
    def __init__(self, wh, mean, ignore_id, min=.5, max=2., class_incidence=None, class_instances=None,
                 inst_classes=(3, 12, 14, 15, 16, 17, 18), scale_method=lambda scale, wh, size: int(scale * wh)):
        self.wh = wh
        self.min = min
        self.max = max
        self.mean = mean
        self.ignore_id = ignore_id
        self.random_gens = [self._rand_location]
        self.scale_method = scale_method

        if class_incidence is not None and class_instances is not None:
            self.true_random = False
            class_incidence_obj = np.load(class_incidence)
            with open(class_instances, 'rb') as f:
                self.class_instances = pickle.load(f)
            inst_classes = np.array(inst_classes)
            class_freq = class_incidence_obj[inst_classes].astype(np.float32)
            class_prob = 1. / (class_freq / class_freq.sum())
            class_prob /= class_prob.sum()
            self.p_class = {k.item(): v.item() for k, v in zip(inst_classes, class_prob)}
            self.random_gens += [self._gen_instance_box]
            print(f'Instance based random cropping:\n\t{self.p_class}')

    def _random_instance(self, name, W, H):
        def weighted_random_choice(choices):
            max = sum(choices)
            pick = random.uniform(0, max)
            key, current = 0, 0.
            for key, value in enumerate(choices):
                current += value
                if current > pick:
                    return key
                key += 1
            return key

        instances = self.class_instances[name]
        possible_classes = list(set(self.p_class.keys()).intersection(instances.keys()))
        roulette = []
        flat_instances = []
        for c in possible_classes:
            flat_instances += instances[c]
            roulette += [self.p_class[c]] * len(instances[c])
        if len(flat_instances) == 0:
            return [0, W - 1, 0, H - 1]
        index = weighted_random_choice(roulette)
        return flat_instances[index]

    def _gen_instance_box(self, W, H, target_wh, name, flipped):
        wmin, wmax, hmin, hmax = self._random_instance(name, W, H)
        if flipped:
            wmin, wmax = W - 1 - wmax, W - 1 - wmin
        inst_box = [wmin, hmin, wmax, hmax]
        for _ in range(50):
            box = self._rand_location(W, H, target_wh)
            if bb_intersection_over_union(box, inst_box) > 0.:
                break
        return box

    def _rand_location(self, W, H, target_wh, *args, **kwargs):
        try:
            w = np.random.randint(0, W - target_wh + 1)
            h = np.random.randint(0, H - target_wh + 1)
        except ValueError:
            print(f'Exception in RandomSquareCropAndScale: {target_wh}')
            w = h = 0
        # left, upper, right, lower)
        return w, h, w + target_wh, h + target_wh

    def _trans(self, img: pimg, crop_box, target_size, pad_size, resample, blank_value):
        return crop_and_scale_img(img, crop_box, target_size, pad_size, resample, blank_value)

    def __call__(self, example):
        image = example['image']
        scale = np.random.uniform(self.min, self.max)
        W, H = image.size
        box_size = self.scale_method(scale, self.wh, image.size)
        pad_size = (max(box_size, W), max(box_size, H))
        target_size = (self.wh, self.wh)
        crop_fn = random.choice(self.random_gens)
        flipped = example['flipped'] if 'flipped' in example else False
        crop_box = crop_fn(pad_size[0], pad_size[1], box_size, example.get('name'), flipped)
        ret_dict = {
            'image': self._trans(image, crop_box, target_size, pad_size, RESAMPLE, self.mean),
        }
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], crop_box, target_size, pad_size, pimg.NEAREST, self.ignore_id)
        for k in ['image_prev', 'image_next']:
            if k in example:
                ret_dict[k] = self._trans(example[k], crop_box, target_size, pad_size, RESAMPLE,
                                          self.mean)
        if 'depth' in example:
            ret_dict['depth'] = self._trans(example['depth'], crop_box, target_size, pad_size, RESAMPLE_D, 0)
        if 'flow' in example:
            ret_dict['flow'] = crop_and_scale_flow(example['flow'], crop_box, target_size, pad_size, scale)
        return {**example, **ret_dict}


class RandomFlip:
    def _trans(self, img: pimg, flip: bool):
        return img.transpose(pimg.FLIP_LEFT_RIGHT) if flip else img

    def __call__(self, example):
        flip = np.random.choice([False, True])
        ret_dict = {}
        for k in ['image', 'image_next', 'image_prev', 'labels', 'depth']:
            if k in example:
                ret_dict[k] = self._trans(example[k], flip)
        if ('flow' in example) and flip:
            ret_dict['flow'] = flip_flow_horizontal(example['flow'])
        return {**example, **ret_dict}
