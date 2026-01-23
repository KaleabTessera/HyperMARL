# HyperMARL Reproduction Guide

## Table of Contents

1. [Installation](#installation)
   - [Specialisation and Synchronisation Game](#specialisation-and-synchronisation-game)
   - [VMAS Experiments](#vmas-experiments)
   - [JaxMARL Experiments](#jaxmarl-experiments)
2. [Experiments](#experiments)
   - [Specialisation and Synchronisation Game](#1-specialisation-and-synchronisation-game)
   - [Dispersion Experiments](#2-dispersion-experiments)
   - [Navigation Experiments](#3-navigation-experiments)
   - [JaxMARL Experiments](#4-jaxmarl-experiments)
   - [MaMuJoCo Experiments Comparing with HAPPO and Kaleidoscope](#5-mamujoco-experiments-comparing-with-happo-and-kaleidoscope-off-policy)


## Installation

### Specialisation and Synchronisation Game

1. Create and activate a conda environment:
```bash
conda create --name hypermarl_spec_sync python=3.10 -y
conda activate hypermarl_spec_sync
```

2. Install required packages:
```bash
pip install -r requirements/spec_game_jax.txt
```


### VMAS Experiments

1. Create and activate a conda environment:
```bash
conda create --name hypermarl_vmas python=3.10 -y
conda activate hypermarl_vmas
```

2. Install required packages:
```bash
pip install -r requirements/requirements_vmas_cpu.txt
```

3. For rendering:
```bash
conda install conda-forge::libglu
```

4. Add Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

If you want to run on GPU, install the following:
```bash
pip install --upgrade "jax[cuda]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### JaxMARL Experiments

1. Create and activate a conda environment:
```bash
conda create --name hypermarl_jaxmarl python=3.10 -y && conda activate hypermarl_jaxmarl
```

2. Install required packages:
```bash
pip install -r requirements/jaxmarl_cpu.txt
pip install flax==0.9.0
# pip install orbax  # to save models
```

If you want to run on GPU, install the following:
```bash
pip install -U "jax[cuda12]==0.4.33"
```     
<!-- pip install --upgrade "jax[cuda]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -->

## Experiments

For all add Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

### 1. Specialisation and Synchronisation Game

1. Run baseline experiments:
```bash
exps/raw_scripts/spec_sync_game/specialisation.sh
exps/raw_scripts/spec_sync_game/synchronisation.sh
```
2. Run HyperMARL version:
```bash
exps/raw_scripts/spec_sync_game/specialisation_hypermarl.sh
```

### 2. Dispersion Experiments
Run the following scripts to reproduce the Dispersion results:
```bash
# baselines
exps/raw_scripts/dispersion/4ag/ippo_ff_nps.sh
exps/raw_scripts/dispersion/4ag/ippo_ff_fups.sh
exps/raw_scripts/dispersion/4ag/mappo_ff_nps.sh
exps/raw_scripts/dispersion/4ag/mappo_ff_fups.sh

# HyperMARL
exps/raw_scripts/dispersion/4ag/ippo_ff_hypernets_linear.sh
exps/raw_scripts/dispersion/4ag/ippo_ff_hypernets_mlp_embedding.sh
exps/raw_scripts/dispersion/4ag/mappo_ff_hypernets_linear.sh
exps/raw_scripts/dispersion/4ag/mappo_ff_hypernets_mlp_embedding.sh

```

### 3. Navigation Experiments

Run experiments to reproduce Navigation results.

#### Same Goals for All Agents & Different Goals for All Agents
```bash
# 2 agents
exps/raw_scripts/navigation/2ag/ippo_linear_hypernets.sh
exps/raw_scripts/navigation/2ag/ippo_mlp_hypernets.sh
exps/raw_scripts/navigation/2ag/ippo_nps.sh
exps/raw_scripts/navigation/2ag/ippo_fups.sh

# 4 agents
exps/raw_scripts/navigation/4ag/ippo_linear_hypernets.sh
exps/raw_scripts/navigation/4ag/ippo_mlp_hypernets.sh
exps/raw_scripts/navigation/4ag/ippo_nps.sh
exps/raw_scripts/navigation/4ag/ippo_fups.sh

# 8 agents
exps/raw_scripts/navigation/8ag/ippo_linear_hypernets.sh
exps/raw_scripts/navigation/8ag/ippo_mlp_hypernets.sh
exps/raw_scripts/navigation/8ag/ippo_nps.sh
exps/raw_scripts/navigation/8ag/ippo_fups.sh
```

#### Half the Agents Share Goals
```bash
# 4 agents (2 share goals)
exps/raw_scripts/navigation/4ag/share_half/ippo_fups.sh
exps/raw_scripts/navigation/4ag/share_half/ippo_linear_hypernets.sh
exps/raw_scripts/navigation/4ag/share_half/ippo_mlp_hypernets.sh
exps/raw_scripts/navigation/4ag/share_half/ippo_nps.sh

# 8 agents (4 share goals)
exps/raw_scripts/navigation/8ag/share_half/ippo_fups.sh
exps/raw_scripts/navigation/8ag/share_half/ippo_linear_hypernets.sh
exps/raw_scripts/navigation/8ag/share_half/ippo_mlp_hypernets.sh
exps/raw_scripts/navigation/8ag/share_half/ippo_nps.sh
```   

DiCo Baselines:

Follow the installation instructions in the [DiCo repository](https://github.com/proroklab/ControllingBehavioralDiversity) and copy and run the following commands in the DiCO folder - `exps/raw_scripts/navigation/dico/2agents.sh` and `exps/raw_scripts/navigation/dico/4agents.sh` and `exps/raw_scripts/navigation/dico/8agents.sh`.


### 4. JaxMARL Experiments

##### Basline experiments:
```bash
exps/raw_scripts/jaxmarl/2s3z/ippo_rnn_baseline.sh
exps/raw_scripts/jaxmarl/2s3z/mappo_rnn_baseline.sh

exps/raw_scripts/jaxmarl/3s5z/ippo_rnn_baseline.sh
exps/raw_scripts/jaxmarl/3s5z/mappo_rnn_baseline.sh

exps/raw_scripts/jaxmarl/smacv2/10ag/ippo_rnn_baseline.sh
exps/raw_scripts/jaxmarl/smacv2/10ag/mappo_rnn_baseline.sh

exps/raw_scripts/jaxmarl/smacv2/20ag/ippo_rnn_baseline.sh
exps/raw_scripts/jaxmarl/smacv2/20ag/mappo_rnn_baseline.sh
```

##### HyperMARL (MLP) for JaxMARL
```bash
exps/raw_scripts/jaxmarl/2s3z/ippo_rnn_hypernets_mlp.sh
exps/raw_scripts/jaxmarl/2s3z/mappo_rnn_hypernets_mlp.sh

exps/raw_scripts/jaxmarl/3s5z/ippo_rnn_hypernets_mlp.sh
exps/raw_scripts/jaxmarl/3s5z/mappo_rnn_hypernets_mlp.sh

exps/raw_scripts/jaxmarl/smacv2/10ag/ippo_rnn_hypernets_mlp.sh
exps/raw_scripts/jaxmarl/smacv2/10ag/mappo_rnn_hypernets_mlp.sh

exps/raw_scripts/jaxmarl/smacv2/20ag/ippo_rnn_hypernets_mlp.sh
exps/raw_scripts/jaxmarl/smacv2/20ag/mappo_rnn_hypernets_mlp.sh
```


### 5. MaMuJoCo Experiments comparing with HAPPO and Kaleidoscope (off-policy)

We forked a version of the HARL repository, which can be found [here](). Including installation and running instructions.
