import argparse
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext, dirname, abspath
from os import makedirs
from matplotlib import cm as colourmaps


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='summer_r',
                          save_path=None,
                          predicted_label='Predicted label',
                          true_label='True label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    original_cm = cm
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(
            range(original_cm.shape[0]), range(original_cm.shape[1])):
        plt.text(
            j,
            i,
            format(original_cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(true_label)
    plt.xlabel(predicted_label)
    if save_path:
        plt.savefig(save_path, format=splitext(save_path)[1][1:])
    plt.close('all')


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
    plot_confusion_matrix(
        cm,
        classes=args.classes,
        normalize=True,
        title=args.title,
        cmap=args.colour,
        save_path=args.o,
        predicted_label=args.pl,
        true_label=args.tl)

if __name__ == '__main__':
    main()
