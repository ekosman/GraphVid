import itertools
import time
from os import path
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
import cv2


def generate_colors(n_colors):
    return np.random.randint(0, 255, size=(n_colors, 3))


colors = generate_colors(2000)


def draw_frame(frame, segments):
    mask = np.zeros_like(frame)
    for i, j in itertools.product(range(segments.shape[0]), range(segments.shape[1])):
        mask[i, j, :] = colors[segments[i, j]]

    blended = cv2.addWeighted(frame, 0.6, mask, 0.4, 0.0)
    return blended


if __name__ == '__main__':
    # img = astronaut()
    clip_length = 16
    cap = cv2.VideoCapture(path.join(r'/home/eitank/Downloads/WhatsApp Video 2021-07-19 at 16.22.43.mp4'))

    # Read until video is completed
    clip = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, fx=1, fy=1, dsize=None)

        # Display the resulting frame

        clip.append(frame)

        if len(clip) == clip_length:
            continue

        clip = np.stack(clip)
        segments = slic(clip, n_segments=500, compactness=8)

        idxs = np.unique(segments)
        for idx in idxs:
            pixel_idxs = np.where(segments == idx)
            colors = clip[pixel_idxs]
            avg_color = colors.mean(axis=0)
            clip[pixel_idxs] = avg_color

        # res = draw_frame(frame, segments)
        for frame in clip:
            # time.sleep(0.5)
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        clip = []

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
