import logging
import click
import configparser
import fnmatch
import re
import decimal
from enum import Enum
from multiprocessing import cpu_count
from os import makedirs, walk
from matplotlib import cm
from os.path import abspath, join, isfile, basename, dirname, realpath, splitext

from deepspectrum.backend.plotting import PLOTTING_FUNCTIONS
from deepspectrum.tools.label_parser import LabelParser
from deepspectrum.backend.extractor import KerasExtractor, PytorchExtractor
from deepspectrum.tools.path import get_relative_path

max_np = cpu_count()

decimal.getcontext().prec = 6

log = logging.getLogger(__name__)


def _check_positive(ctx, param, value):
    if value is None:
        return value
    ivalue = int(value)
    if ivalue <= 0:
        raise click.BadParameter("%s is an invalid positive int value" % value)
    return ivalue


class Filetypes(Enum):
    AUDIO = ['wav', 'ogg', 'flac', 'mp3']
    IMAGE = ['png', 'jpg']


GENERAL_OPTIONS = [
    click.argument(
        "input",
        type=click.Path(dir_okay=True,
                        file_okay=True,
                        exists=True,
                        readable=True),
    ),
    click.option(
        "-c",
        "--config",
        type=click.Path(readable=True, dir_okay=False),
        help=
        "Path to configuration file which specifies available extraction networks. If this file does not exist a new one is created and filled with the standard settings.",
        default=join(dirname(realpath(__file__)), "deep.conf"),
    ),
    click.option(
        "-np",
        "--number-of-processes",
        type=click.IntRange(1, max_np, clamp=True),
        help=
        "Define the number of processes used in parallel for the extraction. If None defaults to cpu-count",
        default=max_np,
    ),
]

PARSER_OPTIONS = [
    click.option(
        "-p",
        "--parser",
        type=click.Path(readable=True, dir_okay=False),
        help=
        "Path to auDeep parser file.",
        default=None)
        
]

PLOTTING_OPTIONS = [
    click.option(
        "-s",
        "--start",
        help=
        "Set a start time from which features should be extracted from the audio files.",
        type=decimal.Decimal,
        default=0,
    ),
    click.option(
        "-e",
        "--end",
        help=
        "Set a end time until which features should be extracted from the audio files.",
        type=decimal.Decimal,
        default=None,
    ),
    click.option(
        "-t",
        "--window-size-and-hop",
        help=
        "Extract deep spectrum features from windows with specified length and hopsize in seconds.",
        nargs=2,
        type=decimal.Decimal,
        default=[None, None],
    ),
    click.option(
        "-nfft",
        default=None,
        help="specify the size for the FFT window in number of samples",
        type=int,
    ),
    click.option(
        "-cm",
        "--colour-map",
        default="viridis",
        help="define the matplotlib colour map to use for the spectrograms",
        type=click.Choice([m for m in cm.cmap_d])),  # ,
    # choices=sorted([m for m in cm.cmap_d]))
    click.option(
        "-fql",
        "--frequency-limit",
        type=int,
        help=
        "define a limit for the frequency axis for plotting the spectrograms",
        default=None,
    ),
    click.option(
        "-sr",
        "--sample-rate",
        type=int,
        help=
        "define a target sample rate for reading the audio files. Audio files will be resampled to this rate before spectrograms are extracted.",
        default=None,
    ),
    click.option(
        "-so",
        "--spectrogram-out",
        help=
        "define an existing folder where spectrogram plots should be saved during feature extraction. By default, spectrograms are not saved on disk to speed up extraction.",
        default=None,
    ),
    click.option(
        "-wo",
        "--wav-out",
        help=
        "Convenience function to write the chunks of audio data used in the extraction to the specified folder.",
        default=None,
    ),
    click.option(
        "-m",
        "--mode",
        help="Type of plot to use in the system.",
        default="spectrogram",
        type=click.Choice(PLOTTING_FUNCTIONS.keys()),
    ),
    click.option(
        "-nm",
        "--number-of-melbands",
        type=int,
        callback=_check_positive,
        help="Number of melbands used for computing the melspectrogram.",
        default=128,
    ),
    click.option(
        "-fs",
        "--frequency-scale",
        help=
        "Scale for the y-axis of the plots used by the system. Defaults to 'chroma' in chroma mode.",
        default="linear",
        type=click.Choice(["linear", "log", "mel"]),
    ),
    click.option(
        "-d",
        "--delta",
        callback=_check_positive,
        help=
        "If given, derivatives of the given order of the selected features are displayed in the plots used by the system.",
        default=None,
    ),
    click.option(
        "-ppdfs",
        "--pretty_pdfs",
        is_flag=True,
        help=
        "Add if you want to create nice pdf plots of the spectrograms the system uses. For figures in your papers ^.^",
    ),
]

EXTRACTION_OPTIONS = [
    click.option(
        "-en",
        "--extraction-network",
        help=
        "specify the CNN that will be used for the feature extraction. You need to specify a valid weight file in .npy format in your configuration file for this network.",
        default="vgg16",
    ),
    click.option(
        "-fl",
        "--feature-layer",
        default="fc2",
        help="name of CNN layer from which to extract the features.",
    ),
    click.option(
        "-bs",
        "--batch-size",
        type=int,
        help=
        "Maximum batch size for feature extraction. Adjust according to your gpu memory size.",
        default=128,
    ),
]

WRITER_OPTIONS = [
    click.option(
        "-o",
        "--output",
        help=
        "The file which the features are written to. Supports csv and arff formats",
        required=True,
        type=click.Path(writable=True, dir_okay=False),
    ),
    click.option(
        "-nl",
        "--no-labels",
        is_flag=True,
        help="Do not write class labels to the output.",
    ),
    click.option(
        "-nts",
        "--no-timestamps",
        is_flag=True,
        help="Remove timestamps from the output.",
    ),
    click.option(
        "-tc",
        "--time-continuous",
        is_flag=True,
        help=
        'Set labelling of features to timecontinuous mode. Only works in conjunction with "-t" and a label file with a matching timestamp column.',
    ),
]

LABEL_OPTIONS = [
    click.option(
        "-lf",
        "--label-file",
        help=
        "csv file with the labels for the files in the form: 'filename, label'. If nothing is specified here or under -labels, the name(s) of the directory/directories are used as labels.",
        default=None,
        type=click.Path(exists=True, dir_okay=False, readable=True),
    ),
    click.option(
        "-el",
        "--explicit-label",
        type=str,
        nargs=1,
        help="Define an explicit label for the input files.",
        default=None,
    ),
]


class Configuration:
    """
    This class handles the configuration of the deep spectrum extractor by reading commandline options and the
    configuration file. It then parses the labels for the audio files and configures the Caffe Network used for
    extraction.
    """

    def __init__(
            self,
            plotting=True,
            extraction=True,
            writer=True,
            parser=False,
            file_type=Filetypes.AUDIO,
            input=None,
            config="deep.conf",
            number_of_processes=max_np,
            colour_map="viridis",
            mode="mel",
            frequency_scale="linear",
            delta=None,
            frequency_limit=None,
            nfft=None,
            start=0,
            end=None,
            window_size_and_hop=None,
            number_of_melbands=128,
            spectrogram_out=None,
            wav_out=None,
            pretty_pdfs=False,
            extraction_network="vgg16",
            feature_layer="fc7",
            batch_size=128,
            output=None,
            time_continuous=False,
            label_file=None,
            explicit_label=None,
            no_timestamps=False,
            no_labels=False,
            sample_rate=None,
            label_dict=None,
            labels=None,
    ):

        self.input_folder = input if not isfile(input) else dirname(input)
        self.config = config
        self.number_of_processes = number_of_processes
        self.model_weights = "imagenet"
        self.file_type = file_type
        self.plotting = plotting
        self.plotting_args = {}
        self.extraction = extraction
        self.extraction_args = {}
        self.writer = writer
        self.writer_args = {}
        self.backend = "keras"
        self.parser = parser

        if self.plotting:
            self.plotting_args["cmap"] = colour_map
            self.plotting_args["mode"] = mode
            self.plotting_args["scale"] = frequency_scale
            self.plotting_args["delta"] = delta
            self.plotting_args["ylim"] = frequency_limit
            self.plotting_args["nfft"] = nfft
            self.plotting_args["start"] = start
            self.plotting_args["end"] = end
            self.plotting_args["window"] = (window_size_and_hop[0]
                                            if window_size_and_hop else None)
            self.plotting_args["hop"] = (window_size_and_hop[1]
                                         if window_size_and_hop else None)
            self.plotting_args["resample"] = sample_rate
            self.plotting_args["base_path"] = self.input_folder
            if self.plotting_args["mode"] == "mel":
                self.plotting_args["melbands"] = number_of_melbands
            if self.plotting_args["mode"] == "chroma":
                self.plotting_args["scale"] = "chroma"
            self.plotting_args["output_spectrograms"] = (
                abspath(spectrogram_out)
                if spectrogram_out is not None else None)
            self.plotting_args["output_wavs"] = (abspath(wav_out) if
                                                 wav_out is not None else None)
            if pretty_pdfs:
                self.plotting_args["file_type"] = "pdf"
                self.plotting_args["labelling"] = True
        if self.extraction:
            self.net = extraction_network
            self.extraction_args["layer"] = feature_layer
            self.extraction_args["batch_size"] = batch_size

        self._load_config()
        self.files = self._find_files(input)
        
        if not self.files:
            log.error(
                f"No files were found under the path {input}. Check the specified input path."
            )
            exit(1)
        
        
        if self.writer:
            self.label_file = label_file
            self.writer_args["output"] = output
            makedirs(dirname(abspath(self.writer_args["output"])),
                     exist_ok=True)
            self.writer_args["continuous_labels"] = (
                ("window" in self.plotting_args) and time_continuous
                and self.label_file)
            self.writer_args["labels"] = explicit_label
            self.writer_args["write_timestamps"] = (
                window_size_and_hop !=
                (None, None)) and not no_timestamps and self.plotting
            self.writer_args["no_labels"] = no_labels

            log.info("Parsing labels...")
            if self.parser:
                self.writer_args["label_dict"] = label_dict
                self.writer_args["labels"] = labels
                self._files_to_extract(relative_paths_in_label_dict=False)
            elif self.label_file is not None:
                self._read_label_file()
            else:
                self._create_labels_from_folder_structure()            

    def _find_files(self, folder):
        log.debug(f'Input file types are "{self.file_type.value}".')
        if isfile(folder) and splitext(folder)[1][1:] in self.file_type.value:
            log.debug(f"{folder} is a single {self.file_type.value}-file.")
            return [folder]

        input_files = []
        for file_type in self.file_type.value:
            globexpression = "*." + file_type
            reg_expr = re.compile(fnmatch.translate(globexpression),
                                  re.IGNORECASE)
            log.debug(f"Searching {folder} for {file_type}-files.")
            for root, dirs, files in walk(folder, topdown=True):
                new_files = [
                    join(root, j) for j in files if re.match(reg_expr, j)
                ]
                log.debug(
                    f"Found {len(new_files)} {file_type}-files in {root}.")
                input_files += new_files
        log.debug(
            f"Found a total of {len(input_files)} {self.file_type.value}-files."
        )
        return sorted(input_files)

    def _files_to_extract(self, relative_paths_in_label_dict=True):
        file_names = set(
            map(
                lambda f: get_relative_path(
                    f, prefix=self.input_folder), self.files))
        if not relative_paths_in_label_dict:
            self.writer_args["label_dict"] = {get_relative_path(
                    key, prefix=self.input_folder): value for key, value in self.writer_args["label_dict"].items()}
        # check if labels are missing for specific files
        
        missing_labels = file_names.difference(self.writer_args["label_dict"])
        if missing_labels:
            log.info(
                f"No labels for: {len(missing_labels)} files. Only processing files with labels."
            )
            self.files = [
                file for file in self.files
                if get_relative_path(
                    file, prefix=self.input_folder) in self.writer_args["label_dict"]
            ]
        log.info(f'Extracting features for {len(self.files)} files.')

    def _read_label_file(self):
        """
        Read labels from either .csv or .tsv files
        :return: Nothing
        """
        if self.label_file.endswith(".tsv"):
            parser = LabelParser(
                self.label_file,
                delimiter="\t",
                timecontinuous=self.writer_args["continuous_labels"],
            )
        else:
            parser = LabelParser(
                self.label_file,
                delimiter=",",
                timecontinuous=self.writer_args["continuous_labels"],
            )

        parser.parse_labels()

        self.writer_args["label_dict"] = parser.label_dict
        self.writer_args["labels"] = parser.labels
        
        self._files_to_extract()

        

    def _create_labels_from_folder_structure(self):
        """
        If no label file is given, either explicit labels or the folder structure is used as class values for the input.
        :return: Nothing
        """
        if self.writer_args["labels"] is None:
            self.writer_args["label_dict"] = {
                get_relative_path(
                    f, prefix=self.input_folder): [basename(dirname(f))]
                for f in self.files
            }
        else:
            # map the labels given on the commandline to all files in a given folder to all input files
            self.writer_args["label_dict"] = {
                get_relative_path(f, prefix=self.input_folder):
                [str(self.writer_args["labels"])]
                for f in self.files
            }
        labels = sorted(
            list(map(lambda x: x[0], self.writer_args["label_dict"].values())))

        self.writer_args["labels"] = [("class", set(labels))]

    def _load_config(self):
        """
        Parses the configuration file given on the commandline. If it does not exist yet, creates a new one containing
        standard settings.
        :param conf_file: configuration file to parse or create
        :return: Nothing
        """
        conf_parser = configparser.ConfigParser()

        # check if the file exists and parse it
        if isfile(self.config):
            log.info("Found config file " + self.config)
            conf_parser.read(self.config)
            main_conf = conf_parser["main"]
            self.plotting_args["size"] = int(main_conf["size"])
            self.backend = main_conf["backend"]
            filetypes = Enum(
                'ConfigurationFiletypes', {
                    'AUDIO': main_conf['audioFormats'].split(','),
                    'IMAGE': main_conf['imageFormats'].split(',')
                })
            self.file_type = filetypes[self.file_type.name]
            if self.extraction:
                keras_net_conf = conf_parser["keras-nets"]
                pytorch_net_conf = conf_parser["pytorch-nets"]
                if self.net in keras_net_conf:
                    self.extractor = KerasExtractor
                    self.extraction_args["weights_path"] = keras_net_conf[
                        self.net]
                    self.extraction_args["model_key"] = self.net
                elif self.net in pytorch_net_conf:
                    self.extractor = PytorchExtractor
                    self.extraction_args["model_key"] = self.net
                else:
                    log.error(
                        f"No model weights defined for {self.net} in {self.config}"
                    )
                    exit(1)
                

        # if not, create it with standard settings
        else:
            log.info("Writing standard config to " + self.config)

            makedirs(dirname(abspath(self.config)), exist_ok=True)
            # Read the defaul config file included in the package
            conf_parser.read(join(dirname(realpath(__file__)), "deep.conf"))
            with open(self.config, "w") as configfile:
                conf_parser.write(configfile)
                log.error(
                    f"Please initialize your configuration file in {self.config}"
                )
                exit(1)
