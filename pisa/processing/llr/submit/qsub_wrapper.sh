#!/bin/bash
#
# Simple wrapper class to submit to a cluster that utilizes the PBS system
# (e.g. PSU and MSU). Upon successful submission, the job file is moved to the
# "submitted/" subdirectory.
#
# NOTE: This is not a generic tool. This
# should be modified for your own usage and convenience.

submitted_count=0
for file in $*
do
    echo $file
    echo "  qsub $file"
	succeeded=""
	while [ -z "$succeeded" -a -f $file ]
	do
		qsub $file && (mv $file submitted/;succeeded="true") || (echo "Failed to submit $file waiting 30 min to try again"; sleep 1800)
	done
	submitted_count=$(( submitted_count + 1 ))
done

echo "Submitted $submitted_count jobs (apparently) successfully."