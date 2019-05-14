from multiprocessing import cpu_count
from deepspectrum.__main__ import cli
from click.testing import CliRunner
from os.path import dirname, join
from shutil import rmtree

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_features_file_level():
    rmtree('/tmp/deepspectrumtest', ignore_errors=True)
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-c',
                               '/tmp/deepspectrumtest/deep.conf', '-o',
                               '/tmp/deepspectrumtest/features.csv'
                           ])
    assert result.exit_code == 1
    assert 'Please initialize your configuration file' in result.output
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               '/tmp/deepspectrumtest/features.csv', '-so',
                               '/tmp/deepspectrumtest/spectrograms', '-r',
                               '/tmp/deepspectrumtest/reduced.csv', '-en',
                               'vgg16', '-sr', 16000, '-m', 'mel', '-fs',
                               'mel', '-c', '/tmp/deepspectrumtest/deep.conf'
                           ])
    assert result.exit_code == 0
    assert 'Done' in result.output


def test_features_time_continuous():
    rmtree('/tmp/deepspectrumtest', ignore_errors=True)
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'twilight', '-o',
                               '/tmp/deepspectrumtest/features-tc.csv', '-en',
                               'densenet201', '-sr', 16000, '-m', 'chroma',
                               '-t', '1', '1', '-tc', '-s', 0, '-e', '2',
                               '-lf',
                               join(
                                   examples,
                                   'labels',
                                   'time-continuous.csv',
                               ), '-fl', 'avg_pool'
                           ])
    assert result.exit_code == 0
    assert 'Done' in result.output
