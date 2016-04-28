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
    qsub $file
    echo "  mv $file submitted/"
    `mv $file submitted/`
done

echo "DONE!"