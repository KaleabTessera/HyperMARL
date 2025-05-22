#!/bin/bash
python baselines/IPPO/ippo_rnn_smax_with_eval.py -m  MAP_NAME=smacv2_10_units SEED=30,1,42,72858,2300658 LR=0.001 +EXP_TAGS=[IPPO,RNN,Baseline,Table3]