#!/bin/bash
python baselines/IPPO/ippo_ff_nps.py --config-name ippo_ff_nps_vmas_navigation.yaml -m SEED=30,1,42,72858,2300658 EXP_NAME=dico_comparison EXP_TAGS=[IPPO,FF,NPS,v2.3,Dico,Part2,Table2] +EVAL_PARALLEL=True env.ENV_KWARGS.agents_with_same_goal=1,4 env.ENV_KWARGS.collisions=False env.ENV_KWARGS.n_agents=4
