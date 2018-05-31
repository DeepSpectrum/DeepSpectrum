#!/bin/bash


basepath=/home/maurice/Desktop/audio-with-new-name
for instance in $basepath/audio/train/*/; do
	echo extract_ds_features -f $instance -t 1 0.1 --no_labels --tc -mode mel -o $basepath/features/train/$(basename $instance).arff
	pipenv run extract_ds_features -f $instance -t 1 0.1 --no_labels --tc -mode mel -o $basepath/features-new/train/$(basename $instance).arff
done;

for instance in $basepath/audio/devel/*/; do
	echo extract_ds_features -f $instance -t 1 0.1 --no_labels --tc -mode mel -o $basepath/features/devel/$(basename $instance).arff
	pipenv run extract_ds_features -f $instance -t 1 0.1 --no_labels --tc -mode mel -o $basepath/features-new/devel/$(basename $instance).arff
done;

