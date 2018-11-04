from setuptools import setup, find_packages

setup(name='deep_spectrum_extractor',
      version='0.2',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'ds-features = deep_spectrum_extractor.cli.ds_extract:main',
              'ds-reduce = deep_spectrum_extractor.tools.feature_reduction:main',
              'ds-svm = deep_spectrum_extractor.learn.scikit_models:main',
              'ds-dnn = deep_spectrum_extractor.learn.tf.dnn.__main__:main',
              'ds-rnn = deep_spectrum_extractor.learn.tf.rnn.__main__:main',
              'ds-cm = deep_spectrum_extractor.tools.performance_stats:main',
              'ds-image-features = deep_spectrum_extractor.cli.image_features:main',
              'ds-plot = deep_spectrum_extractor.cli.ds_plot:main',
              'ds-help = deep_spectrum_extractor.cli.ds_help:main'
          ]
      },
      install_requires=['numpy', 'scipy', 'pandas', 'imread', 'pysoundfile', 'tqdm', 'matplotlib', 'opencv-python', 'librosa', 'sklearn', 'liac-arff'],
      extras_require={'tensorflow-gpu': ['tensorflow-gpu'], 'tensorflow': ['tensorflow']},
      zip_safe=False
      )
