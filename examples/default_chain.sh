#!/usr/bin/env bash

# This example shell script will run each stage of the PISA chain
# by seperately calling each stage. The output will be written to
# a JSON file in each stage. To run this example, from the main
# PISA directory:
#
#   > default_chain.sh

function check(){
  if [ $? != 0 ]; then
    echo "Stage \"$1\" failed - aborting!"
    exit 1;
  fi
}

echo "Running flux stage..."
Flux.py -vvv
check "Flux"

echo "Running oscillations stage..."
Oscillation.py flux.json -vvv
check "Oscillations"

echo "Running effective area stage..."
Aeff.py osc_flux.json -vvv
check "Effective Area"

echo "Running reconstruction stage..."
Reco.py event_rate.json -vvv
check "Reconstruction"

echo "Running PID stage..."
PID.py reco.json -vvv
check "PID"
