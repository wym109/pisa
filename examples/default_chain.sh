#!/bin/sh

# This example shell script will run each stage of the PISA chain
# by seperately calling each stage. The output will be written to
# a JSON file in each stage. To run this example, from the main
# PISA directory:
#
#   > source setup.sh
#   > default_chain.sh

function check(){
  if [ $? != 0 ]; then
    echo "Stage \"$1\" failed - aborting!"
    exit 1;
  fi
}

echo "\nRunning flux stage..."
Flux.py -vvv
check "Flux"

echo "\nRunning oscillations stage..."
Oscillation.py flux.json -vvv
check "Oscillations"

echo "\nRunning trigger stage..."
EventRate.py osc_flux.json $PISA/resources/events/V15_weighted_aeff.hdf5 -vvv
check "Trigger"

echo "\nRunning reconstruction stage..."
Reco.py event_rate.json $PISA/resources/events/V15_weighted_aeff.hdf5 -vvv
check "Reconstruction"

echo "\nRunning PID stage..."
PID.py reco.json $PISA/resources/pid/V15_pid.json -vvv
check "PID"
