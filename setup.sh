#!/bin/sh
echo "Setting up PISA..."
export PISA=$( cd "$( dirname "$0" )" && pwd )
export PYTHONPATH=$PYTHONPATH:$PISA
export PATH=$PATH:$PISA/flux
