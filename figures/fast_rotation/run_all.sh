#!/bin/bash

mkdir -p renders/fast_rotation
python figures/fast_rotation/gt.py
python figures/fast_rotation/onfly.py
python figures/fast_rotation/precomp.py