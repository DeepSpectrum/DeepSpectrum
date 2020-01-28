#!/bin/bash
# Copyright (C) 2020 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, Björn Schuller
#
# This file is part of DeepSpectrum.
#
# DeepSpectrum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSpectrum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DeepSpectrum. If not, see <http://www.gnu.org/licenses/>.
verbose_option="-v"


export PYTHONUNBUFFERED=1

taskName="ComParE2020_Mask"

# base directory for audio files
audio_base="../"
output_base="../features"
melBands="128"

# good choices can be "magma", "plasma", "viridis" or "cividis"
colourMap="magma"
scale="mel"

# Parser for the data set
parser="audeep.backend.parsers.compare20_mask.Compare20MaskParser"

# DeepSpectrum supports the following pre-trained CNN networks: vgg16, vgg19, resnet50, inception_resnet_v2, 
# xception, densenet121, densenet169, densenet201, mobilenet, mobilenet_v2, nasnet_large, nasnet_mobile, alexnet, 
# squeezenet, googlenet.
extractionNetwork="resnet50"
output="$output_base/$taskName.DeepSpectrum_$extractionNetwork.csv"

# recommended feature layer for all pre-trained CNNs except for vgg16, vgg19, and alexnet: "avg_pool"
# recommended feature layer for vgg16, vgg19, and alexnet: "fc2"
featureLayer="avg_pool"

cmd="deepspectrum $verbose_option features-with-parser ../ --parser $parser -o $output -en $extractionNetwork -fl $featureLayer -nm $melBands -fs $scale -cm $colourMap --no-labels"
echo $cmd
$cmd
