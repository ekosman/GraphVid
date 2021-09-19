import cProfile
import sys
import time
from itertools import product
from os import path
from enum import Enum

import matplotlib.pyplot as plt
import torch
from torch import tensor

import cv2
import numpy as np
from fast_slic.avx2 import SlicAvx2
from networkx import DiGraph
from numba import jit, njit, prange
from numpy.linalg import norm
from numpy import unique, stack, asarray, concatenate
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


SPATIAL = 0
TEMPORAL = 1


class EmptyGraphException(Exception):
    pass


def get_adjacents_v3(segments):
    idxs = np.unique(segments)
    return [[x, y] for x, y in product(idxs, idxs) if x != y]


# @jit(nopython=False, parallel=True)
def get_adjacents_v2(segments):
    contour_y = np.where(segments[1:, :] != segments[:-1, :])
    contour_x = np.where(segments[:, 1:] != segments[:, :-1])

    # dy edges
    xs = contour_y[1]

    labels_source = segments[contour_y[0], xs]
    labels_targets = segments[contour_y[0] + 1, xs]

    edges_y = set()
    for s, t in zip(labels_source, labels_targets):
        edges_y.add(frozenset((s, t)))

    # dx edges
    ys = contour_x[0]

    labels_source = segments[ys, contour_x[1]]
    labels_targets = segments[ys, contour_x[1] + 1]

    edges_x = set()
    for s, t in zip(labels_source, labels_targets):
        edges_y.add(frozenset((s, t)))

    return edges_y.union(edges_x)


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


@jit(nopython=True, parallel=True)
def indices_for_segments(superpixels, num_superpixels):
    """
    Retrieves the x,y coordinates of every pixel that correspond to each segment
    :param superpixels:
    :param num_superpixels:
    :return:
    """
    binSize = np.zeros(num_superpixels, dtype=np.int32)
    for i in range(superpixels.shape[0]):
        for j in range(superpixels.shape[1]):
            binSize[superpixels[i, j]] += 1

    # Put the pixels location in the right bin
    result = [(np.empty(binSize[i], dtype=np.int32), np.empty(binSize[i], dtype=np.int32)) for i in
              range(num_superpixels)]
    binPos = np.zeros(num_superpixels, dtype=np.int32)
    for i in range(superpixels.shape[0]):
        for j in range(superpixels.shape[1]):
            binIdx = superpixels[i, j]
            tmp = result[binIdx]
            cellBinPos = binPos[binIdx]
            tmp[0][cellBinPos] = i
            tmp[1][cellBinPos] = j
            binPos[binIdx] += 1

    return result


def get_attr_for_segment_v2(frame, pixel_ys, pixel_xs, method='mean_color'):
    if method == 'mean_color':
        colors = frame[pixel_ys, pixel_xs]
        return (colors * 1.0).mean(axis=0)
    elif method == 'mean_coordinates':
        return np.mean(pixel_ys) / frame.shape[0], np.mean(pixel_xs) / frame.shape[1]
    elif method == 'mean_coordinates_and_mean_color':
        colors = frame[pixel_ys, pixel_xs, :]
        return (colors * 1.0).mean(axis=0), stack([pixel_ys, pixel_xs]).mean(1) / frame.shape[:-1]
    else:
        raise NotImplementedError


# @jit(nopython=True, parallel=True)
def get_attr_for_segment(frame, segments, segment_id, method='mean_color'):
    pixel_idxs = np.array(np.where(segments == segment_id))

    if method == 'mean_color':
        colors = frame[pixel_idxs]
        return (colors * 1.0).mean(axis=0)
    elif method == 'mean_coordinates':
        return np.mean(pixel_idxs[0]) / frame.shape[0], np.mean(pixel_idxs[1]) / frame.shape[1]
    elif method == 'mean_coordinates_and_mean_color':
        colors = frame[pixel_idxs[0], pixel_idxs[1], :]
        return (colors * 1.0).mean(axis=0), pixel_idxs.mean(axis=1) / frame.shape[:-1]
    else:
        raise NotImplementedError


@jit(nopython=True, parallel=True)
def get_attr_for_segment_v3(frame, segments, idxs, method='mean_color'):
    binSize = np.zeros(len(idxs), dtype=np.int32)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            binSize[segments[i, j]] += 1

    pixel_idxs = [np.zeros((2, binSize[idx]), dtype=np.int32) for idx in idxs]
    binPos = np.zeros(len(idxs), dtype=np.int32)
    for y in range(segments.shape[0]):
        for x in range(segments.shape[1]):
            curBin = segments[y, x]
            pixel_idxs[curBin][0, binPos[curBin]] = y
            pixel_idxs[curBin][1, binPos[curBin]] = x
            binPos[curBin] += 1

    if method == 'mean_color':
        colors_per_idx = np.zeros((len(idxs), 3))
        for i, (ys, xs) in enumerate(pixel_idxs):
            color_sum = np.zeros(3, dtype=np.int32)
            for y, x in zip(ys, xs):
                color_sum += frame[y, x]

            color = color_sum / len(ys)

            for c in range(len(color)):
                colors_per_idx[i, c] = color[c]

        return colors_per_idx
    elif method == 'mean_coordinates':
        coords_per_idx = np.zeros((len(idxs), 2))

        for i, (ys, xs) in enumerate(pixel_idxs):
            coords_per_idx[i] = ys.mean() / frame.shape[0], xs.mean() / frame.shape[1]

        return coords_per_idx
    elif method == 'mean_coordinates_and_mean_color':
        colors_coords_per_idx = np.zeros((len(idxs), 5))
        for i, (ys, xs) in enumerate(pixel_idxs):
            color_sum = np.zeros(3, dtype=np.int32)
            for y, x in zip(ys, xs):
                color_sum += frame[y, x]

            color = color_sum / len(ys)

            colors_coords_per_idx[i, 0] = color[0]
            colors_coords_per_idx[i, 1] = color[1]
            colors_coords_per_idx[i, 2] = color[2]

            colors_coords_per_idx[i, 3] = ys.mean() / frame.shape[0]
            colors_coords_per_idx[i, 4] = xs.mean() / frame.shape[1]

        return colors_coords_per_idx
    else:
        raise NotImplementedError


# @jit(nopython=True, parallel=True)
def get_edges_between_frames(sources, targets, k_nearest=5):
    try:
        features_sources = stack(sources[:, 0])
        features_targets = stack(targets[:, 0])
    except Exception as e:
        pass
    # features_sources = color.rgb2lab(features_sources)
    # features_targets = color.rgb2lab(features_targets)
    coordinates_sources = stack(sources[:, 1])
    coordinates_targets = stack(targets[:, 1])

    features_distances = cdist(features_sources, features_targets)
    coordinates_distances = cdist(coordinates_sources, coordinates_targets)

    nearest_neighbors = np.argsort(coordinates_distances, axis=-1)[:, :k_nearest]

    color_dist_of_nearest_neighbors = features_distances.take(nearest_neighbors)

    index_of_nearest_color = color_dist_of_nearest_neighbors.argmin(axis=-1)
    index_of_nearest_color_and_neighbor = nearest_neighbors[range(len(nearest_neighbors)), index_of_nearest_color]
    minimum_distances = color_dist_of_nearest_neighbors.min(axis=-1)
    color_dist_threshold = minimum_distances.mean(axis=-1) + minimum_distances.std(axis=-1)
    mask = minimum_distances < color_dist_threshold

    return index_of_nearest_color_and_neighbor, mask


def only_coords(attrs, coords):
    return coords


def only_attrs(attrs, coords):
    return attrs


def concat(attrs, coords):
    return np.concatenate([attrs, coords])


def node_feature_retriever_definer(node_features_mode):
    if node_features_mode == 'only_coords':
        return only_coords
    elif node_features_mode == 'only_attrs':
        return only_attrs
    elif node_features_mode == 'concat':
        return concat
    else:
        raise NotImplementedError


def create_superpixels_flow_graph(clip, n_segments, compactness, node_features_mode='concat'):
    layer_nodes = [torch.tensor([])] * clip.shape[0]
    layer_edges = [torch.tensor([])] * clip.shape[0]  # spatial edges
    layer_edge_attrs = [torch.tensor([])] * clip.shape[0]  # spatial edges attrs

    temporal_edges = []
    temporal_edge_attrs = []

    # Cluster Map
    all_segments = [SlicAvx2(num_components=n_segments, compactness=compactness, num_threads=10).iterate(frame)
                    for frame in clip]

    for i_frame, frame in enumerate(clip):
        segments = all_segments[i_frame]
        idxs = unique(segments)
        if len(idxs) < 2:
            # plt.Figure()
            # plt.imshow(frame)
            # plt.show()
            layer_nodes[i_frame] = None
            continue

        edges = get_adjacents_v3(segments)
        # edges = get_adjacents_v2(segments)
        superpixels_idxs = indices_for_segments(segments, len(idxs))
        node_attrs_coordinates = [get_attr_for_segment_v2(frame,
                                                          *superpixels_idxs[idx],
                                                          method='mean_coordinates_and_mean_color')
                                  for idx in idxs]

        # nodes = [((i_frame, idx), {'x': node_attrs_coordinates[idx][0]}) for idx in idxs]
        # parent_graph.add_nodes_from(nodes)

        layer_edges[i_frame] = tensor([[u, v] for u, v in edges])
        layer_edge_attrs[i_frame] = tensor(
            [norm(node_attrs_coordinates[u][1] - node_attrs_coordinates[v][1], ord=2) for u, v in edges])
        # for u, v in edges:
        #     d = norm(node_attrs_coordinates[u][1] - node_attrs_coordinates[v][1], ord=2)
        #     parent_graph.add_edge(u, v, edge_attr=d)
        #     parent_graph.add_edge(v, u, edge_attr=d)

        layer_nodes[i_frame] = node_attrs_coordinates

    for idx_prev, idx_cur, prev_nodes, cur_nodes in zip(range(0, len(layer_nodes) - 1), range(1, len(layer_nodes)),
                                                        layer_nodes[:-1], layer_nodes[1:]):
        if prev_nodes is not None and cur_nodes is not None:
            nearest_neighbors, mask = get_edges_between_frames(asarray(prev_nodes), asarray(cur_nodes), k_nearest=10)
            edges = [((idx_prev, source), (idx_cur, neighbor)) for source, neighbor in enumerate(nearest_neighbors) if mask[source]]

            attrs = [norm(prev_nodes[source][1] - cur_nodes[neighbor][1], ord=2) for source, neighbor in enumerate(nearest_neighbors) if mask[source]]

            temporal_edges += edges
            temporal_edge_attrs += attrs

    node_indices = np.cumsum([0] + [len(x) if x is not None else 0 for x in layer_nodes[:-1]])
    x = torch.stack([tensor(node[0]) for nodes in layer_nodes if nodes is not None for node in nodes])

    spatial_edge_attr = torch.cat(layer_edge_attrs)
    spatial_edge_index = torch.cat([idxs + n_nodes for idxs, n_nodes in zip(layer_edges, node_indices)])

    spatial_relations = [SPATIAL] * len(spatial_edge_attr)

    temporal_edge_attrs = torch.tensor(temporal_edge_attrs)
    temporal_edge_index = torch.tensor([[node_indices[src_frame] + src_sp, node_indices[dst_frame] + dst_sp] for (src_frame, src_sp), (dst_frame, dst_sp) in temporal_edges])
    temporal_relations = [TEMPORAL] * len(temporal_edge_attrs)

    res = Data(x=x,
               edge_index=torch.cat([spatial_edge_index, temporal_edge_index]).T.long(),
               edge_attr=torch.cat([spatial_edge_attr, temporal_edge_attrs]),
               edge_type=torch.tensor(spatial_relations + temporal_relations)
               )
    if res.num_nodes == 0:
        raise EmptyGraphException

    return res


class VideoClipToSuperPixelFlowGraph:
    def __init__(self, n_segments=80, compactness=10):
        self.n_segments = n_segments
        self.compactness = compactness

    def __call__(self, clip):
        # clip = torch.transpose(clip, dim0=1, dim1=2)

        # profiler = cProfile.Profile()
        # profiler.enable()
        res = create_superpixels_flow_graph(clip.contiguous().numpy().astype(np.uint8), self.n_segments,
                                            self.compactness)
        # profiler.disable()
        # profiler.print_stats(sort='tottime')

        return res


class NetworkxToGeometric:
    def __call__(self, g):
        start_time = time.time()
        g = from_networkx(g)
        print(f"TIME: {time.time() - start_time}")
        return g


if __name__ == '__main__':
    clip_length = 16
    cap = cv2.VideoCapture(
        path.join(r'/media/eitank/disk2T/Datasets/kinetics-downloader/dataset/train/arranging_flowers/0a8l_Pou_C8.mp4'))

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
