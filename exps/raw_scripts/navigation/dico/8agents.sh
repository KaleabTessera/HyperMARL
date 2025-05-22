# note you need to run these in dico repo
# 8 agents

# 8 agents diff goals sweep
# model.desired_snd=-1,1.2,2.4,4.8 experiment.lr=0.00005,0.0005,0.00025 algorithm.clip_epsilon=0.1,0.2

# 8 agents (4 goals) sweep 
# model.desired_snd=-1,1.2,2.4 experiment.lr=0.00005,0.0005,0.00025 algorithm.clip_epsilon=0.1,0.2

python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py -m model.desired_snd=-1 task.n_agents=8 task.agents_with_same_goal=1 seed=30,1,42,72858,2300658 experiment.render=False

python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py -m model.desired_snd=0 task.n_agents=8 task.agents_with_same_goal=8 seed=30,1,42,72858,2300658 experiment.render=False

python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py -m model.desired_snd=-1 task.n_agents=8 task.agents_with_same_goal=4 seed=30,1,42,72858,2300658 experiment.render=False