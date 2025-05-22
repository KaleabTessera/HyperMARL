#!/bin/bash
python baselines/MAPPO/mappo_rnn_smax_with_eval.py -m  MAP_NAME=smacv2_20_units SEED=30,1,42,72858,2300658 +USE_AGENT_ID_EMBEDDINGS=True LR=0.0003 +EXP_TAGS=[MAPPO,RNN,Baseline,Table3]