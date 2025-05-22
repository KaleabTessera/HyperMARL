#!/bin/bash
python baselines/IPPO/ippo_rnn_smax_mlp_hypernets_eval.py -m  +USE_AGENT_ID_EMBEDDINGS=True +HYPERNET_EMBEDDING_DIM=64 +HYPERNET_HIDDEN_DIMS=[16] LR=0.004 +NUM_SEEDS=5 MAP_NAME=3s5z