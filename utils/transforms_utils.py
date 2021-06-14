from torchvision.transforms import transforms

from transforms import transforms_video


def build_transforms(video_height=112, video_width=112):
    """
    Builds the transformations for video clips
    mean, std and frame dimensions are taken from 'Video classification' in:
                https://pytorch.org/docs/stable/torchvision/models.html

                All pre-trained models expect input images normalized in the
                same way, i.e. mini-batches of 3-channel RGB videos of shape
                (3 x T x H x W), where H and W are expected to be 112, and T
                is a number of video frames in a clip. The images have to be
                loaded in to a range of [0, 1] and then normalized using
                mean = [0.43216, 0.394666, 0.37645] and std = [0.22803, 0.22145, 0.216989].

    Returns: Torchvision transform object

    """
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.RandomResizedCropVideo(size=(video_height, video_width), crop=video_width),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    return res
