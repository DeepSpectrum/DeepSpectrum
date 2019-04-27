DESCRIPTION_EXTRACT = 'Extract deep spectrum features from wav files.'
DESCRIPTION_IMAGE_FEATURES = 'Extract CNN-descriptors from images.'
DESCRIPTION_PLOT = 'Create plots from wav files.'
DESCRIPTION_SCIKIT = 'Train and evaluate optimized scikit-learn models.'
DESCRIPTION_REDUCE= 'Reduce a list of feature files by removing zero features.'
DESCRIPTION_NN = 'Interface for training and evaluating a {}.'
DESCRIPTION_RESULTS = 'Tool to inspect, load and export results.'


def main():
    help_string = f'''
|+ + + + + + + + + + + + + + + + + DEEP SPECTRUM TOOLKIT + + + + + + + + + + + + + + + + + +|
|-------------------------------------------------------------------------------------------|
|* The ds-toolkit exposes the following functionalities via command line interfaces:       *|
|*                                                                                         *|
|**** {'ds-features : '+DESCRIPTION_EXTRACT:81} ****|
|**** {'ds-image-features : '+DESCRIPTION_IMAGE_FEATURES:81} ****|
|**** {'ds-plot : '+DESCRIPTION_PLOT:81} ****|
|**** {'ds-reduce : '+DESCRIPTION_REDUCE:81} ****|
|*                                                                                         *|
|* For each tool, detailed usage descriptions are available with: \'ds-[tool] --help\'       *|
|-------------------------------------------------------------------------------------------|'''

    print(help_string)


if __name__ == '__main__':
    main()
