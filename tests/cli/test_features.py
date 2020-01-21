from multiprocessing import cpu_count
from deepspectrum.__main__ import cli
from click.testing import CliRunner
from os.path import dirname, join
from os import listdir

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_features_file_level(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features',
                               join(examples, 'audio'), '-c',
                               join(tmpdir, 'deep.conf'), '-o',
                               join(tmpdir, 'features.csv')
                           ])
    assert 'Please initialize your configuration file' in result.output
    assert result.exit_code == 1
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               join(tmpdir, 'features.csv'), '-so',
                               join(tmpdir,
                                    'spectrograms'), '-en', 'squeezenet',
                               '-sr', 16000, '-m', 'mel', '-fs', 'mel', '-c',
                               join(tmpdir, 'deep.conf')
                           ])
    print(result.output)
    print(listdir(join(tmpdir, 'spectrograms')))
    assert 'Done' in result.output
    assert result.exit_code == 0


def test_features_file_level_parser(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features-with-parser',
                               join(examples, 'audio'), '-c',
                               join(tmpdir, 'deep.conf'), '-o',
                               join(tmpdir, 'features.csv')
                           ])
    assert 'Please initialize your configuration file' in result.output
    assert result.exit_code == 1
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features-with-parser',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               join(tmpdir, 'features.csv'), '-so',
                               join(tmpdir,
                                    'spectrograms'), '-en', 'squeezenet',
                               '-sr', 16000, '-m', 'mel', '-fs', 'mel', '-c',
                               join(tmpdir, 'deep.conf')
                           ])
    print(result.output)
    print(listdir(join(tmpdir, 'spectrograms')))
    assert 'Done' in result.output
    assert result.exit_code == 0


def test_features_file_level_single_file(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features',
                               join(examples, 'audio', 'dog', '1.flac'), '-np',
                               cpu_count(), '-cm', 'viridis', '-o',
                               join(tmpdir, 'features-single-file.csv'), '-so',
                               join(tmpdir,
                                    'spectrograms'), '-en', 'alexnet', '-sr',
                               16000, '-m', 'mel', '-fs', 'mel', '-fl', 'fc7'
                           ])
    print(result.output)
    print(listdir(join(tmpdir, 'spectrograms')))
    assert 'Done' in result.output
    assert result.exit_code == 0


def test_features_time_continuous(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'features',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'twilight', '-o',
                               join(tmpdir, 'features-tc.csv'), '-en', 'vgg16',
                               '-sr', 16000, '-m', 'chroma', '-t', '1', '1',
                               '-tc', '-s', 0, '-e', '2', '-lf',
                               join(
                                   examples,
                                   'labels',
                                   'time-continuous.csv',
                               ), '-fl', 'fc1'
                           ])
    print(result.output)
    assert 'Done' in result.output
    assert result.exit_code == 0
