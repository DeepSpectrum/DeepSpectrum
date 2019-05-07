#!/bin/bash

# Replace with correct paths
challengePath="/path/to/Extended_DAIC_audiovisual_LLDs_wav"
outputPath="/output/folder"

# Leave as it is
taskName="daic"
taskPath="$outputPath/$taskName"
windowSize="4"; hopSize="1"; mode="mel"; nmel="128"
nets="vgg16 densenet201"
cmaps="viridis"

###################
#one feature file per each audio clip
###################
for n in $nets; do
	for c in $cmaps; do
		for h in $hopSize; do
			wavPath="/home/spa/Downloads/Extended_DAIC_audiovisual_LLDs_wav"
			cd $wavPath
			for f in *.wav; do
				featPath="$taskPath/audio_features_deepspectrum/$h/$taskName-net=$n-winSize=$windowSize-hopSize=$h-cmap=$c-mode=$mode"
				specPath="$taskPath/spectrograms/$h/$taskName-net=$n-winSize=$windowSize-hopSize=$h-cmap=$c-mode=$mode"						
				if [ $n = "vgg16" -o $n = "vgg19" ]
				then
					echo "deepspectrum features $f -t $windowSize $h -en $n -fl fc2 -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv"
					deepspectrum -v features $f -t $windowSize $h -en $n -fl fc2 -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv
				else
					echo "deepspectrum features $f -t $windowSize $h -en $n -fl avg_pool -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv"
					deepspectrum -v features $f -t $windowSize $h -en $n -fl avg_pool -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv
				fi
			done
		done
	done
done
