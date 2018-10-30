from .ds_extract import DESCRIPTION as ds_extract
from .image_features import DESCRIPTION as ds_image_features
from .ds_plot import DESCRIPTION as ds_plot
from ..learn.linear_svm import DESCRIPTION as ds_svm
from ..tools.feature_reduction import DESCRIPTION as ds_reduce
from ..learn.tf.dnn.__main__ import DESCRIPTION as ds_dnn
from ..learn.tf.rnn.__main__ import DESCRIPTION as ds_rnn
from ..tools.performance_stats import DESCRIPTION as ds_cm

def main():
    help_string = f'''
        |+ + + + + + + + + + + + + + + + + DEEP SPECTRUM TOOLKIT + + + + + + + + + + + + + + + + + +|
        |-------------------------------------------------------------------------------------------|
        |* The ds-toolkit exposes the following functionalities via command line interfaces:       *|
        |*                                                                                         *|
        |**** {'ds-features : '+ds_extract:81} ****|
        |**** {'ds-image-features : '+ds_image_features:81} ****|
        |**** {'ds-plot : '+ds_plot:81} ****|
        |**** {'ds-reduce : '+ds_reduce:81} ****|
        |**** {'ds-svm : '+ds_svm:81} ****|
        |**** {'ds-dnn : '+ds_dnn:81} ****|
        |**** {'ds-rnn : '+ds_rnn:81} ****|
        |**** {'ds-cm : '+ds_cm:81} ****|
        |*                                                                                         *|
        |* For each tool, detailed usage descriptions are available with: \'ds-[tool] --help\'       *|
        |-------------------------------------------------------------------------------------------|'''

    print(help_string)

if __name__=='__main__':
    main()