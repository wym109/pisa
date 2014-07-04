#!/bin/sh
echo "Setting up PISA..."
export PISA=$( cd "$( dirname "$0" )" && pwd )
export PYTHONPATH=$PISA:$PYTHONPATH
export PATH=$PISA/flux:$PISA/trigger:$PISA/oscillations:$PISA/reco:$PISA/pid:$PISA/examples:$PATH

