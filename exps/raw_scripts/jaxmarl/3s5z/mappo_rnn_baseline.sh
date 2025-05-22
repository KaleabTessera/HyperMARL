#!/bin/bash
python baselines/MAPPO/mappo_rnn_smax_with_eval.py -m  MAP_NAME=3s5z +USE_AGENT_ID_EMBEDDINGS=True LR=0.002 +EXP_TAGS=[MAPPO,RNN,Baseline,Table3]