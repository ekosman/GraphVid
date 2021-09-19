from torchvision.transforms import transforms

from transforms import transforms_video
from transforms.create_superpixels_flow_graph import VideoClipToSuperPixelFlowGraph, NetworkxToGeometric
from transforms.edge_transforms import AdditiveGaussianEdgeNoise, RandomSpatialEdgesRemoval
from transforms.node_transforms import RandomNodeRemoval, AdditiveGaussianNodeNoise
from transforms.transforms_video import RandomResizedCropVideo


def build_transforms(superpixels=50, train=False):
    """"""

    """
    Builds the transformations for video clips
    Returns: Torchvision transform object
    """
    # mean = [0.43216, 0.394666, 0.37645]
    # std = [0.22803, 0.22145, 0.216989]

    if train:
        res = transforms.Compose([
            RandomResizedCropVideo(size=(480, 640), ),
            VideoClipToSuperPixelFlowGraph(n_segments=superpixels),
            # AdditiveGaussianEdgeNoise(sigma=0.05),
            # RandomSpatialEdgesRemoval(p=0.8),
            # AdditiveGaussianNodeNoise(sigma=0.05),
            # RandomNodeRemoval(),
        ])
    else:
        res = transforms.Compose([
            # transforms_video.ToTensorVideo(),
            # transforms_video.RandomResizedCropVideo(size=(video_height, video_width), crop=video_width),
            # transforms_video.NormalizeVideo(mean=mean, std=std)
            RandomResizedCropVideo(size=(480, 640), ),
            VideoClipToSuperPixelFlowGraph(n_segments=superpixels),
            # NetworkxToGeometric(),
        ])

    return res
