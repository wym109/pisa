#!/bin/sh

# This example shell script will run each stage of the PISA chain
# by seperately calling each stage. The output will be written to
# a JSON file in each stage. To run this example, from the main
# PISA directory:
#
#   > source setup.sh
#   > default_chain.sh

echo "\nRunning flux stage..."
Flux.py -vvv

echo "\nRunning oscillations stage..."
OscillationMaps.py flux.json -vvv

echo "\nRunning trigger stage..."
EventRate.py osc_flux.json $PISA/resources/events/V15_weighted_aeff.hdf5 -vvv

echo "\nRunning reconstruction stage..."
Reco.py event_rate.json $PISA/resources/events/V15_weighted_aeff.hdf5 -vvv

echo "\nRunning PID stage..."
ApplyPID.py reco.json $PISA/resources/pid/V15_pid.json -vvv

