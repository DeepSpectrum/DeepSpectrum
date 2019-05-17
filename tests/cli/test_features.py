from multiprocessing import cpu_count
from deepspectrum.__main__ import cli
from click.testing import CliRunner
from os.path import dirname, join

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_features_file_level(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-c',
                               join(tmpdir, 'deep.conf'), '-o',
                               join(tmpdir, 'features.csv')
                           ])
    assert result.exit_code == 1
    assert 'Please initialize your configuration file' in result.output
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               join(tmpdir, 'features.csv'), '-so',
                               join(tmpdir, 'spectrograms'), '-en', 'vgg16',
                               '-sr', 16000, '-m', 'mel', '-fs', 'mel', '-c',
                               join(tmpdir, 'deep.conf')
                           ])
    assert result.exit_code == 0
    assert 'Done' in result.output


def test_features_file_level_single_file(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio', 'dog', 'dog.flac'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               join(tmpdir, 'features-single-file.csv'), '-so',
                               join(tmpdir, 'spectrograms'), '-en', 'vgg16',
                               '-sr', 16000, '-m', 'mel', '-fs', 'mel'
                           ])
    assert result.exit_code == 0
    assert 'Done' in result.output


def test_features_time_continuous(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'twilight', '-o',
                               join(tmpdir,
                                    'features-tc.csv'), '-en', 'densenet201',
                               '-sr', 16000, '-m', 'chroma', '-t', '1', '1',
                               '-tc', '-s', 0, '-e', '2', '-lf',
                               join(
                                   examples,
                                   'labels',
                                   'time-continuous.csv',
                               ), '-fl', 'avg_pool'
                           ])
    assert result.exit_code == 0
    assert 'Done' in result.output
