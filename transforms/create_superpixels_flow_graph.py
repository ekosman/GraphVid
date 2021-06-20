import cProfile
import sys
from os import path
# import matplotlib.pyplot as plt
import cv2
import networkx as nx
import numpy as np
import torch
from networkx import DiGraph
from scipy.spatial.distance import cdist
from skimage import color
from skimage.segmentation import slic
from torch_geometric.utils import from_networkx


def get_adjacents(segments):
    horizontal_shift_match = segments[:, :-1] != segments[:, 1:]
    vertical_shift_match = segments[-1:, :] != segments[1:, :]

    adjacents = {v: set() for v in np.unique(segments)}

    ys, xs = np.where(horizontal_shift_match)
    try:
        left_segments = segments[ys, np.clip(xs - 1, 0, xs.max())]
        right_segments = segments[ys, np.clip(xs + 1, 0, xs.max())]
    except Exception as e:
        i = 1

    ys, xs = np.where(vertical_shift_match)
    up_segments = segments[np.clip(ys - 1, 0, ys.max()), xs]
    down_segments = segments[np.clip(ys + 1, 0, ys.max()), xs]

    centers = segments[ys, xs]

    adjacents_left, = np.where(left_segments != centers)
    adjacents_right, = np.where(right_segments != centers)
    adjacents_up, = np.where(up_segments != centers)
    adjacents_down, = np.where(down_segments != centers)

    for v, v_adjs in adjacents.items():
        left_idx = list(set(np.where(centers == v)[0]).intersection(adjacents_left))
        left_adjs = left_segments[left_idx]

        right_idx = list(set(np.where(centers == v)[0]).intersection(adjacents_right))
        right_adjs = right_segments[right_idx]

        up_idx = list(set(np.where(centers == v)[0]).intersection(adjacents_up))
        up_adjs = up_segments[up_idx]

        down_idx = list(set(np.where(centers == v)[0]).intersection(adjacents_down))
        down_adjs = down_segments[down_idx]
        v_adjs.update(left_adjs)
        v_adjs.update(right_adjs)
        v_adjs.update(up_adjs)
        v_adjs.update(down_adjs)

    return adjacents


def get_attr_for_segment(frame, segments, segment_id, method='mean_color'):
    pixel_idxs = np.where(segments == segment_id)
    colors = frame[pixel_idxs]

    if method == 'mean_color':
        return (colors * 1.0).mean(axis=0)
    if method == 'mean_coordinates':
        return np.mean(pixel_idxs[0]) / frame.shape[0], np.mean(pixel_idxs[1]) / frame.shape[1]
    else:
        raise NotImplementedError


def get_distances(sources, targets, alpha):
    features_sources = np.stack([source[0] for source in sources])
    features_targets = np.stack([target[0] for target in targets])
    features_sources = color.rgb2lab(features_sources)
    features_targets = color.rgb2lab(features_targets)
    coordinates_sources = np.stack([source[1] for source in sources])
    coordinates_targets = np.stack([target[1] for target in targets])
    features_distances = cdist(features_sources, features_targets)
    coordinates_distances = cdist(coordinates_sources, coordinates_targets)

    return features_distances + alpha * coordinates_distances


def create_superpixels_flow_graph(clip, n_segments=200, compactness=10):
    last_layer_nodes = None
    parent_graph = DiGraph()

    for i_frame, frame in enumerate(clip):
        child_graph = DiGraph()
        segments = slic(frame, n_segments=n_segments, compactness=compactness, start_label=0)
        idxs = np.unique(segments)
        if len(idxs) < 2:
            last_layer_nodes = None
            continue

        adjacents = get_adjacents(segments)
        # idxs.sort()
        node_attrs = {idx:  get_attr_for_segment(frame, segments, idx, method='mean_color') for idx in idxs}
        node_coordinates = {idx:  get_attr_for_segment(frame, segments, idx, method='mean_coordinates') for idx in idxs}
        current_layer_nodes = [(node_attrs[idx], node_coordinates[idx]) for idx in idxs]

        for idx in idxs:
            node_name = f"level_{i_frame}_segment_{idx}"

            if not child_graph.has_node(node_name):
                child_graph.add_node(node_name, x=np.concatenate([node_attrs[idx], node_coordinates[idx]]))

            for adjacent in adjacents[idx]:
                adjacent_name = f"level_{i_frame}_segment_{adjacent}"

                if not child_graph.has_node(adjacent_name):
                    child_graph.add_node(adjacent_name, x=np.concatenate([node_attrs[adjacent], node_coordinates[adjacent]]))

                if not child_graph.has_edge(node_name, adjacent_name):
                    child_graph.add_edge(node_name, adjacent_name)
                if not child_graph.has_edge(adjacent_name, node_name):
                    child_graph.add_edge(adjacent_name, node_name)

        parent_graph = nx.compose(parent_graph, child_graph)

        if last_layer_nodes is not None:
            feature_distances = get_distances(last_layer_nodes, current_layer_nodes, alpha=4000)
            nearest_neighbors = np.argmin(feature_distances, axis=1)
            dists = feature_distances[list(range(feature_distances.shape[0])), nearest_neighbors]
            mean_ = np.mean(dists)
            std = np.std(dists)
            nearest_neighbors = nearest_neighbors[dists < mean_ + std]
            # plt.figure()
            # plt.hist(dists, bins=100)
            # plt.axvline(mean_+std, color='r')
            # plt.show()
            for source, neighbor in enumerate(nearest_neighbors):
                if f"level_{i_frame - 1}_segment_{source}" not in parent_graph.nodes:
                    raise Exception
                if f"level_{i_frame}_segment_{neighbor}" not in parent_graph.nodes:
                    raise Exception
                parent_graph.add_edge(f"level_{i_frame - 1}_segment_{source}",
                                      f"level_{i_frame}_segment_{neighbor}")

        last_layer_nodes = current_layer_nodes

    return parent_graph


class VideoClipToSuperPixelFlowGraph:
    def __init__(self, n_segments=200, compactness=10):
        self.n_segments = n_segments
        self.compactness = compactness

    def __call__(self, clip):
        clip = torch.transpose(clip, dim0=1, dim1=2)
        return create_superpixels_flow_graph(clip, self.n_segments, self.compactness)


class NetworkxToGeometric:
    def __call__(self, g):
        return from_networkx(g)


if __name__ == '__main__':
    clip_length = 16
    cap = cv2.VideoCapture(path.join(r'/media/eitank/disk2T/Datasets/kinetics-downloader/dataset/train/arranging_flowers/0a8l_Pou_C8.mp4'))

    # Read until video is completed
    clip = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, fx=0.5, fy=0.5, dsize=None)
        clip.append(frame)
        if len(clip) == clip_length:
            profiler = cProfile.Profile()
            profiler.enable()
            X = create_superpixels_flow_graph(clip)
            profiler.disable()
            profiler.print_stats(sort='tottime')
            X = from_networkx(X)
            print(f"Size of clip: {sys.getsizeof(clip)}")
            print(f"Size of graph: {sys.getsizeof(X)}")
            clip = []

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
