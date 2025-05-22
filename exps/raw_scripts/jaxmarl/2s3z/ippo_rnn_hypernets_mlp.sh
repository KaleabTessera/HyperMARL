#!/bin/bash
python baselines/IPPO/ippo_rnn_smax_mlp_hypernets_eval.py -m  +USE_AGENT_ID_EMBEDDINGS=True +HYPERNET_EMBEDDING_DIM=4 +HYPERNET_HIDDEN_DIMS=[32] LR=0.004 +NUM_SEEDS=5 MAP_NAME=2s3z