#!/usr/bin/bash

mkdir -p renders/car
# GT
python figures/car/car.py renders/car/gt.exr gt --spp 16384
# GGN18 w/ Adaptive Discretization
python figures/car/car.py renders/car/ggn.exr --lookup sat
# Ours
python figures/car/car.py renders/car/ours.exr ours