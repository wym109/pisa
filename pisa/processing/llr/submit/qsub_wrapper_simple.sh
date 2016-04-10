#!/bin/bash
#
# Simple wrapper class to submit to lionga cluster,
# move job to "submitted/" subdir.
#
# NOTE: This is not a generic tool to be run from any directory. This
# should be modified this for your own usage and convenience.

for file in $*
do
    echo $file
    echo "  qsub $file"
	succeeded=""
	while [ -z "$succeeded" -a -f $file ]
	do
		qsub $file && (mv $file submitted/;succeeded="true") || (echo "Failed to submit $file waiting 30 min to try again"; sleep 1800)
	done
done

echo "Submitted all jobs (apparently) successfully."