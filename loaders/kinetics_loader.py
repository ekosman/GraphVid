import logging
import random

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips
from os import path

from transforms.create_superpixels_flow_graph import EmptyGraphException


class Kinetics(VisionDataset):
    """
    `Kinetics <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

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

    def __init__(self, dataset_path, frames_per_clip=16, step_between_clips=8, steps_between_frames=1,
                 extensions=('avi', 'mp4', 'mkv'), transform=None, _precomputed_metadata=None,
                 num_workers=8, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0, **kwargs):
        super(Kinetics, self).__init__(dataset_path)

        self.show_errors = False
        classes = list(sorted(list_dir(dataset_path)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.steps_between_frames = steps_between_frames
        self.total_clip_duration_in_frames = frames_per_clip * steps_between_frames
        video_list = [x[0] for x in self.samples]
        precomputed_metadata_path = path.join(dataset_path, 'metadata.pt')
        flag_metadata_exists = False
        if path.exists(precomputed_metadata_path):
            _precomputed_metadata = torch.load(precomputed_metadata_path)
            flag_metadata_exists = True
        self.video_clips = VideoClips(
            video_list,
            self.total_clip_duration_in_frames,
            step_between_clips,
            None,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )
        self.transform = transform

        if not flag_metadata_exists:
            torch.save(self.video_clips.metadata, precomputed_metadata_path)

        self.cached_graphs = [None] * len(self)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def get_item_aux(self, idx):
        if self.cached_graphs[idx] is not None:
            return self.cached_graphs[idx]

        # video is Tensor[T, H, W, C]
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video = video[range(0, self.total_clip_duration_in_frames, self.steps_between_frames)]
        video = (video * 1.0).to(torch.float32)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        self.cached_graphs[idx] = video, label

        if video is None or label is None:
            pass
        return video, label

    def __getitem__(self, idx):
        """
        Method to access the i'th sample of the dataset
        Args:
            index: the sample index

        Returns: the i'th sample of this dataset
        """
        succ = False
        while not succ:
            try:
                data = self.get_item_aux(idx)
                succ = True
            except EmptyGraphException as e:
                idx = random.randint(0, len(self) - 1)

                if self.show_errors:
                    logging.warning(e)

        return data
