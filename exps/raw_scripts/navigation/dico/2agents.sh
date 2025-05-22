# note you need to run these in dico repo
# 2 agents 
# Use original hyparams from open source implementation - https://github.com/proroklab/ControllingBehavioralDiversity/tree/b6fc469e3ab14f8fd79b1bc2ad3dbf3948587a39 

python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py -m model.desired_snd=1.2 task.agents_with_same_goal=1 seed=30,1,42,72858,2300658

python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py -m model.desired_snd=0 task.agents_with_same_goal=2 seed=30,1,42,72858,2300658