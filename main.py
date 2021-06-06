import itertools
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

    cap = cv2.VideoCapture(path.join(r'F:\Downloads', r'yt1s.com - Canon Rock Jerry C cover by Laura.mp4'))

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, fx=0.4, fy=0.4, dsize=None)

        if ret:
            # Display the resulting frame
            segments = slic(frame, n_segments=1500, compactness=10)
            res = draw_frame(frame, segments)

            cv2.imshow('Frame', res)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
