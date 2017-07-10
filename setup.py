from setuptools import setup

setup(name='deep_spectrum_extractor',
      version='0.1.1',
      packages=['deep_spectrum_extractor', 'deep_spectrum_extractor.cli', 'deep_spectrum_extractor.tools',
                'deep_spectrum_extractor.backend'],
      entry_points={
          'console_scripts': [
              'extract_ds_features = deep_spectrum_extractor.cli.__main__:main',
              'reduce_ds_features = deep_spectrum_extractor.tools.feature_reduction:main'
          ]
      },
      install_requires=['numpy', 'scipy', 'imread', 'pysoundfile', 'tqdm', 'matplotlib', 'tqdm'],
      zip_safe=False
      )
