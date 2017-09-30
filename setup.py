from setuptools import setup, find_packages

setup(name='deep_spectrum_extractor',
      version='0.1.2',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'extract_ds_features = deep_spectrum_extractor.cli.__main__:main',
              'reduce_ds_features = deep_spectrum_extractor.tools.feature_reduction:main'
          ]
      },
      install_requires=['numpy', 'scipy', 'imread', 'pysoundfile', 'tqdm', 'matplotlib', 'opencv-python', 'librosa'],
      extras_require={'tensorflow-gpu': ['tensorflow-gpu'], 'tensorflow': ['tensorflow']},
      zip_safe=False
      )
