from collections import OrderedDict
from subprocess import Popen, PIPE
import matplotlib.animation as manimation

import cv2


def get_movie_info(cv_ptr_to_movie):
    fps = cv_ptr_to_movie.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cv_ptr_to_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return {"fps": fps, "frame_count": frame_count, "duration": duration, "duration_ms": duration * 1000}


def verify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        info = get_movie_info(cap)
        return True
    except ZeroDivisionError:
        return False


def get_frames_in_specific_time(cv_ptr_to_movie, t_vec):
    info = get_movie_info(cv_ptr_to_movie)
    duration = info["duration"]
    fps = info["fps"]
    total_frames = fps * duration

    frames = OrderedDict()
    for t in t_vec:
        frame_number = t / duration * total_frames
        cv_ptr_to_movie.set(1, frame_number)
        ret, frame = cv_ptr_to_movie.read()
        frames[t] = frame

    return frames


def save_frames_to_folder(folder, pre_string, extension, frames):
    for key, value in frames.items():
        if value is not None:
            image_name = f"{folder}/{pre_string}_{key}.{extension}"
            print(f"Save image {image_name}")
            cv2.imwrite(f"{image_name}", value)
        else:
            print(f"Skip image {image_name}")


class MovieClip:
    """
    Iterator class for a video
    yields frames from the specified start time until the specified end time with frame_stride intervals between frames
    """
    def __init__(self, cv_cap, start_time, end_time, frame_stride=4):
        """
        Args:
            cv_cap: The opencv instance of VideoCapture
            start_time: the exact time of the frame in the corresponding video to start the iteration (seconds)
            end_time: the exact time of the frame in the corresponding video to stop the iteration (seconds)
            frame_stride: stride between frames (n_frames)
        """

        self.cv_cap = cv_cap
        self.start_time_ms = start_time * 1000
        self.end_time_ms = end_time * 1000
        self.frame_stride = frame_stride
        try:
            self.info = get_movie_info(cv_cap)
        except ZeroDivisionError:
            self.info = None
        self.frame_i = None

    def is_video_ok(self):
        return self.info['fps'] != 0 if self.info is not None else False

    def is_start_time_valid(self):
        return self.start_time_ms <= self.info['duration_ms']

    def is_end_time_valid(self):
        return self.end_time_ms < self.info['duration_ms']

    def __iter__(self):
        self.frame_i = -1
        self.cv_cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time_ms)
        return self

    def _finished(self):
        return self.cv_cap.get(cv2.CAP_PROP_POS_MSEC) > self.end_time_ms

    def __next__(self):
        while True:
            if self._finished():
                raise StopIteration

            self.frame_i += 1
            ret, frame = self.cv_cap.read()
            if not ret:
                raise StopIteration

            if self.frame_i % self.frame_stride == 0:
                return frame, self.cv_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def convert_2_gif(src, target):
    """
    Converts a movie to gif. Requires ffmpeg installed.
    Args:
        src: path to the movie
        target: path to the target file (should contain .gif extension)

    Returns: The output of the ffmpeg process (read from stdout)

    """
    output = Popen(['ffmpeg', '-y', '-i', src, target],
                   stdout=PIPE).stdout.read().decode(encoding='utf-8')

    return output


def get_movie_writer(title, artist, comment, fps):
    """
    Creates a movie writer for matplotlib figures.
    Once created, you can create matplotlib figures, plot content as you desire and grab it by: writer.grab_frame()
    Args:
        title: title used for the video metadata
        artist: artist used for the video metadata
        comment: comment used for the video metadata
        fps: fps of the output video

    Returns: A pointer to a newly created movie writer

    """
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist=artist,
                    comment=comment)
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer