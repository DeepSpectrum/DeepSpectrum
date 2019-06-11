from click.testing import CliRunner
from deepspectrum.__main__ import cli
from multiprocessing import cpu_count
from os.path import join, dirname

cur_dir = dirname(__file__)
examples = join(dirname(dirname(cur_dir)), 'examples')


def test_plot(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli,
                           args=[
                               '-vv', 'plot',
                               join(examples, 'audio'), '-np',
                               cpu_count(), '-cm', 'twilight', '-so',
                               join(tmpdir, 'pretty-spectrograms'), '-sr',
                               16000, '-m', 'mel', '-fs', 'spectrogram', '-fs',
                               'log', '-ppdfs', '-d', '1', '-wo',
                               join(tmpdir, 'wav-chunks'), '-t', '1', '1',
                               '-fql', '12000'
                           ])
    assert 'Done' in result.output
    assert result.exit_code == 0
