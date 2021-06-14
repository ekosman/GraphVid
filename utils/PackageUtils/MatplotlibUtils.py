import logging
import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as sklearnTSNE
from MulticoreTSNE import MulticoreTSNE as UlyanovTSNE


def plot_scatter_with_labels(points, labels, show_fig=False, file_path=None, close_fig=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('gist_rainbow')
    classes = np.unique(labels)
    n_classes = len(classes)
    ax.set_prop_cycle(get_color_cycler(cmap='gist_rainbow', n_colors=n_classes))

    for label in classes:
        idx, = np.where(labels == label)
        plt.plot(points[idx, 0], points[idx, 1], label=str(label), linestyle="None", marker='.')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if show_fig:
        plt.show()

    if file_path is not None:
        plt.savefig(file_path)

    if close_fig:
        plt.close()


def plot_graph_to_file(x, y, file_name, title=None, x_label=None, y_label=None):
    """
    Plots the specified data to a figure and saves it
    Args:
        x: (optional) data for x-axis
        y: data for y-axis
        title: (optional) title of the graph
        x_label: (optional) label of the x axis
        y_label: (optional) label of the y axis
        file_name: path of the file to save
    """
    plt.figure()
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    if x:
        plt.plot(x, y)
    else:
        plt.plot(y)
    plt.savefig(file_name)
    plt.close()


def plot_histogram_to_file(values, file_name, bins=5000, title=None, x_label=None, y_label=None):
    """
    Plots a histogram of the specified data to a figure and saves it
    Args:
        values: the data to create histogram for
        file_name: path of the file to save
        bins: the amount of bins in the histogram
        title: (optional) title of the graph
        x_label: (optional) label of the x axis
        y_label: (optional) label of the y axis
    """
    plt.figure()
    plt.hist(values, bins=bins)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.close()


def get_color_cycler(cmap='gist_rainbow', n_colors=10):
    """
    returns a cycler for using more than 10 colors with matplotlib
    list of color maps is available at: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    Args:
        cmap: the name of the color map to use
        n_colors: number of colors to use in the cycler

    Returns: a cycler instance

    """
    cm = plt.get_cmap(cmap)
    return cycler(color=[cm(1. * i / n_colors) for i in range(n_colors)])


def reduce_dims_and_plot(x,
           y=None,
           title=None,
           file_name=None,
           perplexity=30,
           library='Multicore-TSNE',
           perform_PCA=False,
           projected=None,
           figure_type='2d',
           show_figure=False,
           close_figure=True,
           text=None,
           **plt_kwargs):
    """
    Creates a 2D visualization of the data using T-SNE
    Args:
        x: array of (n_samples, n_features) representing the data to visualize

        y: (optional) array of (n_samples,) with a label for each x sample

        title: supply a title for the output plot

        file_name: supply path for saving the output plot

        perplexity: parameter for T-SNE

        library: which library should use for T-SNE.
                    Options are: sklearn | Multicore-TSNE
                    The second is much faster
                    See on github: https://github.com/DmitryUlyanov/Multicore-TSNE

        perform_PCA: (Boolean) Whether perform PCA before T-SNE with 8 singular vectors or not

        projected: Use pre-calculated vectors and and skip T-SNE

        figure_type: which type of figure to use (2d | 3d)

        show_figure: whether call plt.show()

        close_figure: whether call plt.close()

        text: (optional) add text to the figure.
                Dictionary: points |===> text
                for example:
                    {(0.1, 0.3): 'point1',
                     (0.2, 0.1): 'point2'}

        plt_kwargs: additional arguments for plot functions

    Returns: the calculated low-dimensional vectors featured by T-SNE (n_samples, 2)

    """
    if projected is None:
        if perform_PCA:
            logging.info("PCA...")
            x = PCA(n_components=8).fit_transform(x)

        logging.info("TSNE...")
        if library == 'sklearn':
            projected = sklearnTSNE(n_components=2, verbose=2, perplexity=perplexity).fit_transform(x)
        elif library == 'Multicore-TSNE':
            projected = UlyanovTSNE(n_components=2, verbose=2, perplexity=perplexity, n_jobs=8, n_iter=600).fit_transform(x)
        else:
            raise NotImplementedError(f'Can\'t find implementation of TSNE using {library}')

        logging.info("TSNE done!")

    fig = plt.figure()

    if y is not None:
        cm = plt.get_cmap('gist_rainbow')
        classes = np.unique(y)
        n_classes = len(classes)
        if figure_type == '2d':
            ax = fig.add_subplot(111)
            ax.set_prop_cycle(get_color_cycler(cmap='gist_rainbow', n_colors=n_classes))
            for c in classes:
                idx, = np.where(y == c)
                plt.plot(projected[idx, 0], projected[idx, 1], label=str(c), linestyle="None", marker='.', **plt_kwargs)
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        elif figure_type == '3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.set_prop_cycle(get_color_cycler(cmap='gist_rainbow', n_colors=n_classes))
            for i_c, c in enumerate(classes):
                idx, = np.where(y == c)
                ax.plot(projected[idx, 0], projected[idx, 1], np.ones(len(idx))*i_c, label=str(c), linestyle="None", marker='.', **plt_kwargs)
                plt.legend()

            ax.view_init(azim=0, elev=-90)

            if text:
                for point, text_content in text.items():
                    ax.text(*point, text_content)
        else:
            raise NotImplementedError(f"figure type {figure_type} not implemented")

    else:
        plt.plot(projected[:, 0], projected[:, 1], linestyle="None", marker='.')

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    if title is not None:
        plt.title(title)

    if show_figure:
        plt.show()
    if file_name is not None:
        plt.savefig(file_name)
    if close_figure:
        plt.close()

    return projected