from setuptools import setup, find_packages

setup(name='deep_spectrum_extractor',
      version='0.1',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'ds_features = deep_spectrum_extractor.cli.__main__:main',
              'reduce_features = deep_spectrum_extractor.tools.feature_reduction:main'
          ]
      },
      package_data={'': ['*.conf']},
      install_requires=['numpy', 'scipy', 'imread', 'pysoundfile', 'tqdm', 'matplotlib', 'tqdm']
      )
