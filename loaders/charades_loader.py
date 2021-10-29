import logging
import os
from os import path
import pandas as pd
import numpy as np

import torch
from torchvision.datasets.video_utils import VideoClips

from loaders.base_video_dataset import CachingVideoDataset


class Charades(CachingVideoDataset):
    """
    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): the `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip
    """

    def __init__(self, dataset_path, phase='train', frames_per_clip=16, step_between_clips=8, steps_between_frames=1,
                 extensions=('avi', 'mp4', 'mkv'), transform=None, _precomputed_metadata=None,
                 num_workers=8, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0, **kwargs):
        super(Charades, self).__init__(kwargs.get('cache_root'))

        self.annotations_path = path.join(dataset_path, 'annotations', f'Charades_v1_{phase}.csv')
        self.annotations = pd.read_csv(self.annotations_path)
        self.annotations['id'] += r'.mp4'
        self.action_annotations = self.prepare_action_labels(self.annotations)

        self.root = dataset_path
        self.show_errors = False

        with open(path.join(dataset_path, 'annotations', 'Charades_v1_classes.txt'), 'r') as fp:
            lines = fp.read().splitlines(keepends=False)
        self.classes = sorted([line.split(' ')[0] for line in lines])
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.steps_between_frames = steps_between_frames
        self.total_clip_duration_in_frames = frames_per_clip * steps_between_frames
        self.step_between_clips = step_between_clips
        video_list = Charades.find_existing_videos(videos_dir=path.join(dataset_path, 'videos'),
                                                   required_videos=set(self.annotations['id'].values))
        video_list = [path.join(dataset_path, 'videos', video) for video in video_list]
        precomputed_metadata_path = path.join(dataset_path, f'metadata_{phase}.pt')
        flag_metadata_exists = False
        self.fps = 12
        if path.exists(precomputed_metadata_path):
            _precomputed_metadata = torch.load(precomputed_metadata_path)
            flag_metadata_exists = True

        logging.info("Preparing VideoClips...")
        self.video_clips = VideoClips(
            video_list,
            self.total_clip_duration_in_frames,
            step_between_clips,
            frame_rate=self.fps,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )

        logging.info("VideoClips ready!")
        self.transform = transform
        self.path_2_idx = {p: i for i, p in enumerate(self.video_clips.video_paths)}

        if not flag_metadata_exists:
            torch.save(self.video_clips.metadata, precomputed_metadata_path)

    def prepare_action_labels(self, annotations):
        res = dict()

        for index, row in annotations.iterrows():
            video_name = row['id']
            res[video_name] = []
            if pd.isna(row['actions']):
                continue
            actions = row['actions'].split(r';')
            for action in actions:
                label, start, end = action.split(' ')
                start, end = float(start), float(end)
                res[video_name].append((start, end, label))

            res[video_name].sort()

        return res

    @staticmethod
    def find_existing_videos(videos_dir, required_videos):
        existing_videos = set(os.listdir(videos_dir))
        res = existing_videos.intersection(required_videos)
        logging.info(f"Found {len(res)} out of {len(required_videos)} videos")
        return res

    def get_item_aux(self, idx):
        # if self.cached_graphs[idx] is not None:
        #     return self.cached_graphs[idx]

        # video is Tensor[T, H, W, C]
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_name = self.video_clips.video_paths[video_idx].split(r'/')[-1]
        video = video[range(0, self.total_clip_duration_in_frames, self.steps_between_frames)]
        video = (video * 1.0).to(torch.float32)

        clip_start = clip_idx * self.step_between_clips
        clip_end = clip_start + self.total_clip_duration_in_frames
        clip_start, clip_end = clip_start / self.fps, clip_end / self.fps
        action_annotations = self.action_annotations[video_name]
        labels = np.zeros(len(self.classes))
        for does_intersect, label in (
                (check_intersection(action_start, action_end, clip_start, clip_end), self.class_to_idx[label])
                for action_start, action_end, label in action_annotations):
            if does_intersect:
                labels[label] = 1

        if self.transform is not None:
            video = self.transform(video)

        # self.cached_graphs[idx] = video, label

        if video is None or labels is None:
            pass
        return video, torch.tensor(labels)


def check_intersection(window1_start, window1_end, window2_start, window2_end):
    return not (window1_end < window2_start or window2_end < window1_start)


if __name__ == '__main__':
    dataset = Charades(r'/media/eitank/disk2T/Datasets/Charades_v1', phase='test',
                       cache_root=r'/media/eitank/disk2T/Datasets/Charades_v1/cache/test')
    dataset[0]
