import argparse
import numpy as np
import tfplot
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from os.path import splitext, dirname, abspath
from os import makedirs
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm as colourmaps
from sklearn.metrics import confusion_matrix

DESCRIPTION='Create a pdf plot from the textual representation of a confusion matrix.'


def plot_confusion_matrix_from_pred(pred,
                                    true,
                                    classes,
                                    normalize=True,
                                    title='Confusion matrix',
                                    cmap='summer_r',
                                    save_path=None,
                                    predicted_label='Predicted label',
                                    true_label='True label'):
    cm = confusion_matrix(true, pred, labels=classes)
    return plot_confusion_matrix(
        cm,
        classes,
        normalize=normalize,
        title=title,
        cmap=cmap,
        predicted_label=predicted_label,
        true_label=true_label)


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='summer_r',
                          predicted_label='Predicted label',
                          true_label='True label',
                          percentages=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = matplotlib.figure.Figure(dpi=200)
    original_cm = cm
    total_samples = np.sum(original_cm)
    if normalize:
        with np.errstate(divide='ignore',invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, vmin=0, vmax=1, cmap=cmap)
    fig.colorbar(im)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xlabel(predicted_label)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel(true_label)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(
            range(original_cm.shape[0]), range(original_cm.shape[1])):
        text = format(original_cm[i, j], fmt)
        if percentages:
            text += '\n{0:.1%}'.format(original_cm[i,j]/total_samples)
        ax.text(
            j,
            i,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    fig.set_tight_layout(True)
    return fig


def save_fig(fig, save_path):
    canvas = FigureCanvasAgg(fig)
    fig.savefig(save_path, format=splitext(save_path)[1][1:])


def fig_to_tf_summary(fig, tensor_name):
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def main():
    parser = argparse.ArgumentParser(
        description=
        'Create nice looking plot from textual representation of confusion matrix.'
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        type=int,
        help=
        'Input a flattened textual representation of the confusionmatrix (in C order).'
    )
    parser.add_argument(
        '-pl', help='Name for predicted label.', default='Predicted label')
    parser.add_argument(
        '-tl', help='Name for true label.', default='True label')
    parser.add_argument(
        '-title',
        help='Title for confusionmatrix.',
        default='Confusion Matrix')
    parser.add_argument(
        '-colour',
        help='matplotlib colourmap to use.',
        choices=sorted([m for m in colourmaps.cmap_d]),
        default='summer_r')
    parser.add_argument(
        '-o', required=True, help='Output path for confusionmatrix.')
    parser.add_argument(
        '-classes',
        required=True,
        nargs='+',
        help=
        'Name of the classes for the confusion matrix in the order they should appear on the top.'
    )
    args = parser.parse_args()
    number_of_classes = isqrt(len(args.i))
    assert (number_of_classes**2 == len(args.i)), 'Not a quadratic matrix!'
    cm = np.reshape(args.i, (isqrt(len(args.i)), isqrt(len(args.i))))
    assert (number_of_classes == len(args.classes)
            ), 'Invalid combination of confusion matrix and class labels!'
    save_path = abspath(args.o)
    makedirs(dirname(save_path), exist_ok=True)
    fig = plot_confusion_matrix(
        cm,
        classes=args.classes,
        normalize=True,
        title=args.title,
        cmap=args.colour,
        predicted_label=args.pl,
        true_label=args.tl)
    save_fig(fig, args.o)


if __name__ == '__main__':
    main()
