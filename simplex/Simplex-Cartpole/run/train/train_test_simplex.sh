#!/usr/bin/env sh
python main_ips.py --mode 'train' --config ./config/simplex/train/test_simplex.json --id new_simplex_ep5 \
--weights './models/pretrained_best'

