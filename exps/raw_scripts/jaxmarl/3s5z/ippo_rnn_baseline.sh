#!/bin/bash
python baselines/IPPO/ippo_rnn_smax_with_eval.py -m  +NUM_SEEDS=5 MAP_NAME=3s5z LR=0.004 +EXP_TAGS=[IPPO,RNN,Baseline,Table3]