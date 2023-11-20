# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center, corner2center
from pysot.core.config import cfg
import time
import random

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

class SubDataset(object):
    def __init__(self, name, root, anno, num_use, start_idx):
        self.name = name
        self.root = root
        self.anno = anno
        self.num_use = num_use
        self.start_idx = start_idx
        print("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)

        self._filter_bad_video(meta_data)
        for video in list(meta_data.keys()):
            frame_names = meta_data[video]['frame_names']
            tracks = meta_data[video]['tracks']
            if len(frame_names) <= 0 or len(tracks) <= 0:
                print("{} has no frames".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        print("{} loaded".format(self.name))
        self.pick = self.shuffle()

    def _filter_bad_video(self, meta_data):
        for video in list(meta_data.keys()):
            frame_names = meta_data[video]['frame_names']
            tracks = meta_data[video]['tracks']
            initial_frame = frame_names[0]
            is_bad = False
            for trk, bbox_dict in tracks.items():
                if initial_frame not in bbox_dict:
                    is_bad = True
            if is_bad:
                print('Filtering %s' % video)
                del meta_data[video]

    def log(self):
        print("{} start-index {} select [{}/{}]".format(
            self.name, self.start_idx, self.num_use, self.num))

    def shuffle(self):
        pick = []
        while len(pick) < self.num_use:
            lists = list(range(self.start_idx, self.start_idx + self.num))
            random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        image_path = os.path.join(self.root, video, '%s.jpg' % frame)
        if frame in self.labels[video]['tracks'][track]:
            image_anno = self.labels[video]['tracks'][track][frame]
        else:
            image_anno = None
        return image_path, image_anno

    def get_template_and_search_frames(self, index, max_len, max_gap):
        video_name = self.videos[index]
        video = self.labels[video_name]
        video_keys = list(video['tracks'].keys())
        track = video_keys[random.randint(0, len(video_keys) - 1)]

        frame_names = video['frame_names']
        left_max = max(len(frame_names) - max_len, 0)
        left = random.randint(0, left_max)
        resample_cnt = 0
        while self.get_image_anno(video_name, track, frame_names[left])[1] is None:
            left = random.randint(0, left_max)
            resample_cnt += 1
            if resample_cnt > 5:
                left = 0
        right = min(left + max_len, len(frame_names))

        sampled_frames = frame_names[left:right]
        search_frames = []
        for f in sampled_frames:
            img_path, img_anno = self.get_image_anno(video_name, track, f)
            search_frames.append((img_path, img_anno))

        if max_gap == -1:
            template_idx = 0
            template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
        else:
            template_min = max(right - max_gap, 0)

            for sampling_trial in range(10):
                template_idx = random.randint(template_min, left)
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
                if template_anno is not None:
                    break

            if template_anno is None:
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[0])

        if template_anno is None:
            print('Template anno is None', video_name)
            exit(0)
        return search_frames, template_frame, template_anno, video_name

    def get_template_and_search_frames_random_interval(self, index, max_len, max_gap, max_interval):
        video_name = self.videos[index]
        video = self.labels[video_name]
        video_keys = list(video['tracks'].keys())
        track = video_keys[random.randint(0, len(video_keys) - 1)]

        frame_names = video['frame_names']

        avg_interval = max_interval
        while avg_interval * (max_len - 1) + 1 > len(frame_names) and avg_interval > 1:
            avg_interval = avg_interval - 1

        # Sample first test frame
        while True:
            left_max = len(frame_names) - (avg_interval*(max_len-1) + 1)
            left = random.randint(0, left_max)

            if self.get_image_anno(video_name, track, frame_names[left])[1] == None:
                avg_interval = avg_interval - 1
                if avg_interval == 0:
                    left = 0
                    break
            else:
                break

        search_ids = []
        search_ids.append(left)

        # Sample rest of the test frames with random interval
        last = left
        while len(search_ids) < max_len:
            # sample id with interval
            last = last + random.randint(1, max_interval)
            if last > len(frame_names) - 1:
                break
            search_ids.append(last)

        # if length is not enough, randomly sample new ids
        if len(search_ids) < max_len:
            valid_ids = [i for i in range(left, len(frame_names)) if i not in search_ids]
            new_ids = random.choices(valid_ids, k=min(len(valid_ids), max_len - len(search_ids)))
            search_ids = search_ids + new_ids
            search_ids = sorted(search_ids, key=int)

        search_frames = []
        for i in search_ids:
            img_path, img_anno = self.get_image_anno(video_name, track, frame_names[i])
            search_frames.append((img_path, img_anno))

        if max_gap == -1:
            template_idx = 0
            template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
        else:
            right = search_ids[-1] + 1
            template_min = max(right - max_gap, 0)

            for sampling_trial in range(10):
                template_idx = random.randint(template_min, left)
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
                if template_anno is not None:
                    break

            if template_anno is None:
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[0])

        if template_anno is None:
            print('Template anno is None', video_name)
            exit(0)
        return search_frames, template_frame, template_anno, video_name

    def get_template_and_search_frames_uniform_interval(self, index, max_len, max_gap, max_interval):
        video_name = self.videos[index]
        video = self.labels[video_name]
        video_keys = list(video['tracks'].keys())
        track = video_keys[random.randint(0, len(video_keys) - 1)]

        frame_names = video['frame_names']

        interval = random.randint(1, max_interval)
        while interval * (max_len - 1) + 1 > len(frame_names) and interval > 1:
            interval = interval - 1

        # Sample first test frame
        left_max = len(frame_names) - (interval * (max_len - 1) + 1)
        left = random.randint(0, left_max)
        resample_cnt = 0
        while self.get_image_anno(video_name, track, frame_names[left])[1] is None:
            left = random.randint(0, left_max)
            resample_cnt += 1
            if resample_cnt > 5:
                left = 0

        sampled_frames = frame_names[left::interval][:max_len]
        search_frames = []
        for f in sampled_frames:
            img_path, img_anno = self.get_image_anno(video_name, track, f)
            search_frames.append((img_path, img_anno))

        if max_gap == -1:
            template_idx = 0
            template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
        else:
            right = min(left + interval*(max_len - 1), len(frame_names))
            template_min = max(right - max_gap, 0)

            for sampling_trial in range(10):
                template_idx = random.randint(template_min, left)
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[template_idx])
                if template_anno is not None:
                    break

            if template_anno is None:
                template_frame, template_anno = self.get_image_anno(video_name, track, frame_names[0])

        if template_anno is None:
            print('Template anno is None', video_name)
            exit(0)
        return search_frames, template_frame, template_anno, video_name

    def __len__(self):
        return self.num


class SequenceDataset(Dataset):
    def __init__(self, max_len):
        super(SequenceDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        self.max_len = max_len
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use
            self.all_dataset.append(sub_dataset)

            sub_dataset.log()

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            random.shuffle(p)
            pick += p
            m = len(pick)
        print("shuffle done!")
        print("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
            orig_cx, orig_cy = (shape[0] + shape[2]) * 0.5, (shape[1] + shape[3]) * 0.5
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        s_x = s_z * (image.shape[0] / cfg.TRACK.EXEMPLAR_SIZE)

        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = np.array(center2corner(Center(cx, cy, w, h)))
        gt_offset = np.array([orig_cx - s_x * 0.5, orig_cy - s_x * 0.5])
        return bbox, scale_z, gt_offset

    def _data_perturb(self, bbox, boundary):
        # bbox : x1 y1 x2 y2
        cx, cy, w, h = corner2center(bbox)
        # perturbation amount in original image scale
        mu, sigma = cfg.DATASET.PERTURB.MEAN, cfg.DATASET.PERTURB.VAR

        # Add noise to bbox
        cx = cx + np.random.normal(mu, sigma)
        cy = cy + np.random.normal(mu, sigma)
        w = w + np.random.normal(mu, sigma)
        h = h + np.random.normal(mu, sigma)

        # Clip bbox to the image boundary
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        w = max(5, min(w, boundary[1]))
        h = max(5, min(h, boundary[0]))

        x1, x2, y1, y2 = center2corner(Center(cx, cy, w, h))
        return [x1, x2, y1, y2]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        if cfg.DATASET.SEQUENCE.SAMPLE_MODE == 'sequential':
            sampled_frames, template_frame, template_anno, video_name = dataset.get_template_and_search_frames(index,
                                                                                                               self.max_len,
                                                                                                               cfg.TRAIN.TEMPLATE_GAP)

        elif cfg.DATASET.SEQUENCE.SAMPLE_MODE == 'random_interval':
            if random.random() > cfg.DATASET.SEQUENCE.SAMPLE_PROB:
                sampled_frames, template_frame, template_anno, video_name = dataset.get_template_and_search_frames(
                    index,
                    self.max_len,
                    cfg.TRAIN.TEMPLATE_GAP)
            else:
                sampled_frames, template_frame, template_anno, video_name = dataset.get_template_and_search_frames_random_interval(
                    index,
                    self.max_len,
                    cfg.TRAIN.TEMPLATE_GAP,
                    cfg.DATASET.SEQUENCE.MAX_INTERVAL)

        elif cfg.DATASET.SEQUENCE.SAMPLE_MODE == 'uniform_interval':
            if random.random() > cfg.DATASET.SEQUENCE.SAMPLE_PROB:
                sampled_frames, template_frame, template_anno, video_name = dataset.get_template_and_search_frames(
                    index,
                    self.max_len,
                    cfg.TRAIN.TEMPLATE_GAP)
            else:
                sampled_frames, template_frame, template_anno, video_name = dataset.get_template_and_search_frames_uniform_interval(
                    index,
                    self.max_len,
                    cfg.TRAIN.TEMPLATE_GAP,
                    cfg.DATASET.SEQUENCE.MAX_INTERVAL)
        else:
            raise NotImplementedError

        template_frame = cv2.imread(template_frame)
        if cfg.DATASET.PERTURB.PERTURB:
            template_anno = self._data_perturb(template_anno, template_frame.shape[:2])

        # get image
        images = []
        gt_orig_list = []
        for img_path, img_anno in sampled_frames:
            img = cv2.imread(img_path)
            if img_anno is None:
                gt_orig = np.array([0, 0, 0, 0])
            else:
                gt_orig = img_anno.copy()
                if cfg.DATASET.PERTURB.PERTURB:
                    gt_orig = self._data_perturb(gt_orig, img.shape[:2])
            images.append(img)
            gt_orig_list.append(gt_orig)

        num_frames = len(sampled_frames)
        for i in range(self.max_len - num_frames):
            images.append(img.copy())
            gt_orig_list.append(gt_orig.copy())
        return {
            'name': dataset.name,
            'num_frames': num_frames,
            'images': np.array(images),
            'template': template_frame,
            'template_bbox': template_anno,
            'gt_orig': np.array(gt_orig_list),
            'video_name': video_name,
        }
