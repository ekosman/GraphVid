import logging
import os
import random
from abc import ABC, abstractmethod
from os import path
import torch
from collections import Mapping, Container
from sys import getsizeof

from transforms.create_superpixels_flow_graph import EmptyGraphException
from utils.PackageUtils.TorchUtils import compress_data, uncompress_data


class BaseVideoDataset(ABC):
    """
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): the `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip
    """

    def __init__(self, **kwargs):
        super(BaseVideoDataset, self).__init__()
        self.classes = []
        self.video_clips = None
        self.show_errors = None

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    @abstractmethod
    def get_item_aux(self, idx):
        pass

    def __getitem__(self, idx):
        """
        Method to access the i'th sample of the dataset
        Args:
            idx: the sample index

        Returns: the i'th sample of this dataset
        """
        succ = False
        while not succ:
            try:
                data = self.get_item_aux(idx)
                succ = True
            except (EmptyGraphException, AssertionError, IndexError) as e:
                idx = random.randint(0, len(self) - 1)

                if self.show_errors:
                    logging.warning(e)

        return data


def size(tensor):
    return tensor.element_size() * tensor.nelement()


i=1
class CachingVideoDataset(BaseVideoDataset, ABC):
    def __init__(self, cache_root=None, return_name=False, **kwargs):
        super(CachingVideoDataset, self).__init__()
        self.cache_root = cache_root
        self.return_name = return_name
        try:
            if cache_root is not None:
                os.makedirs(cache_root, exist_ok=True)
        except Exception as e:
            logging.info(e)
            raise Exception

    def get_item_properties(self, item):
        """
        Returns video name and clip idx
        :param item:
        :return:
        """
        video_idx, clip_idx = self.video_clips.get_clip_location(item)
        video_path = self.video_clips.video_paths[video_idx]
        _, video_name = video_path.split(os.sep)[-2:]
        video_name = video_name.split('.')[0]
        return video_name, clip_idx, video_path

    def __getitem__(self, item):
        global i
        video_name, clip_idx, video_path = self.get_item_properties(item)
        video_idx = self.path_2_idx[video_path]
        if self.cache_root is not None:
            dump_path = path.join(self.cache_root, f"{video_name}_clip_{clip_idx}.file")
        loaded = False
        if self.cache_root is not None:
            if path.exists(dump_path):
                try:
                    data = torch.load(dump_path)
                    loaded = True
                except (EOFError, RuntimeError, OSError) as e:
                    pass

        if not loaded:
            data = super(CachingVideoDataset, self).__getitem__(item)

        data, label = data
        if not hasattr(data, 'normalized') or not data.normalized:
            data.x /= 255
            data.normalized = True

        data = compress_data(data), label
        if self.cache_root is not None:
            torch.save(data, dump_path)

        # print(i)
        # i += 1

        data, label = data
        data = uncompress_data(data), label

        if self.return_name:
            return data[0], data[1], video_idx

        return data


