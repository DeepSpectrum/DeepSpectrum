from setuptools import setup, find_packages

setup(name='deep_spectrum_extractor',
      version='0.2',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'extract_ds_features = deep_spectrum_extractor.cli.__main__:main',
              'reduce_ds_features = deep_spectrum_extractor.tools.feature_reduction:main',
              'linear_svm = deep_spectrum_extractor.learn.linear_svm:main',
              'dnn = deep_spectrum_extractor.learn.tf.dnn.__main__:main',
              'rnn = deep_spectrum_extractor.learn.tf.rnn.__main__:main',
              'plot_cm = deep_spectrum_extractor.tools.performance_stats:main',
              'image_cnn_features = deep_spectrum_extractor.cli.image_features:main',
              'plot_wavs = deep_spectrum_extractor.cli.create_plots:main'
          ]
      },
      install_requires=['numpy', 'scipy', 'pandas', 'imread', 'pysoundfile', 'tqdm', 'matplotlib', 'opencv-python', 'librosa', 'sklearn', 'liac-arff'],
      extras_require={'tensorflow-gpu': ['tensorflow-gpu'], 'tensorflow': ['tensorflow']},
      zip_safe=False
      )
