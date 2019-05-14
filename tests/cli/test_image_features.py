from click.testing import CliRunner
from deepspectrum.__main__ import cli
from multiprocessing import cpu_count
from os.path import join, dirname
from shutil import rmtree

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_image_features():
    rmtree('/tmp/deepspectrumtest', ignore_errors=True)
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'image-features',
                               join(examples, 'pictures'), '-np',
                               cpu_count(), '-o',
                               '/tmp/deepspectrumtest/image-features.arff',
                               '-en', 'vgg19', '-el', 'justAnimals'
                           ])
    assert 'Total params' in result.output
    assert 'Done' in result.output
    assert result.exit_code == 0
