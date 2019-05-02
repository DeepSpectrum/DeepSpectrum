#!/bin/bash

# Replace with correct paths
challengePath="/path/to/Official_AVEC_SPLITS_v2.1"
outputPath="/output/folder"


# Leave as it is
taskName="avec19som"
taskPath="$outputPath/$taskName"
windowSize="4"; hopSize="1"; mode="mel"; nmel="128"
nets="vgg16 densenet201"
cmaps="viridis"
partitions="Development Train Test"

###################
#one feature file per each audio clip
###################
for n in $nets; do
	for c in $cmaps; do
		for p in $partitions; do
			wavPath="$challengePath/$p/audio"
			cd $wavPath
			for f in *; do
				featPath="$taskPath/audio_features_deepspectrum/$p/$taskName-net=$n-winSize=$windowSize-hopSize=$hopSize-cmap=$c-mode=$mode"
				specPath="$taskPath/spectrograms/$p/$taskName-net=$n-winSize=$windowSize-hopSize=$hopSize-cmap=$c-mode=$mode"						
				if [ $n = "vgg16" -o $n = "vgg19" ]
				then
					echo "deepspectrum features $f -t $windowSize $hopSize -en $n -fl fc2 -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv"
					deepspectrum features $f -t $windowSize $hopSize -en $n -fl fc2 -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv
				else
					echo "deepspectrum features $f -t $windowSize $hopSize -en $n -fl avg_pool -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv"
					deepspectrum features $f -t $windowSize $hopSize -en $n -fl avg_pool -m $mode -nm $nmel -cm $c -nl -o $featPath/${f%.wav}.csv
				fi
			done
		done
	done
done
