#!/bin/sh
echo "Setting up PISA..."
export PISA=$( cd "$( dirname "$0" )" && pwd )
PROB3PATH=$PISA/prob3/Prob3++.20121225/
export PYTHONPATH="${PYTHONPATH}:${PISA}:${PROB3PATH}"
export PATH=$PATH:$PISA/flux
