from click.testing import CliRunner
from deepspectrum.__main__ import cli
from multiprocessing import cpu_count
from os.path import join, dirname
from shutil import rmtree

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_plot():
    rmtree('/tmp/deepspectrumtest', ignore_errors=True)
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-v', 'plot',
                               join(examples, 'wav'), '-np',
                               cpu_count(), '-cm', 'twilight', '-so',
                               '/tmp/deepspectrumtest/pretty-spectrograms',
                               '-sr', 16000, '-m', 'mel', '-fs', 'spectrogram',
                               '-fs', 'log', '-ppdfs', '-d', '1', '-wo',
                               '/tmp/deepspectrumtest/wav-chunks', '-t', '1',
                               '1', '-fql', '12000'
                           ])
    assert 'Done' in result.output
    assert result.exit_code == 0
