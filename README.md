<h1 align="center">
    <p>HyperMARL: Adaptive Hypernetworks for Multi-Agent RL </p>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2412.04233"><img src="https://img.shields.io/badge/arXiv-2412.04233-b31b1b.svg" alt="arXiv"></a>
  <a href="https://colab.research.google.com/github/kaleabtessera/hypermarl/blob/main/quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>



<p align="center">
  <img src="assets/hypermarl_black.gif" width="80%" alt="HyperMARL overview">
</p>

If your prefer a static image, here is the [PNG version](assets/hypermarl.png).

## TL;DR 📜
**🧩 Problem**  
Independent policies (**NoPS**) scale poorly and are sample-inefficient, while fully shared policies (**FuPS**) are efficient but can collapse to uniform behaviour due to cross-agent interference -- which can be exacerbated when a shared network **couples** observations and agent IDs.

**🛠️ Our approach**  
We propose **HyperMARL** -- a MARL architecture that uses *agent-conditioned hypernetworks* to **decouple** observation- and agent-conditioned gradients and dynamically generate *agent-specific* actor and critic parameters. This enables agents to exhibit diverse or homogeneous behaviours as needed, **without altering the RL learning objective, requiring prior knowledge of the optimal diversity or sequential updates**.

---

### Key Features
| 🔑 | Feature  |
|----|---------|
| 🧬 **Agent-conditioned hypernetwork** | A shared hypernetwork generates per‐agent actor and critic parameters on the fly. |
| 🔀 **Gradient decoupling** | Decouples observation- and agent-conditioned gradients, which empirically reduces gradient variance and cross‐agent interference. |
| 📊 **Competitive results** | Matches or is competitive with NoPS, FuPS (+/– ID) and three diversity-promoting baselines **across a wide range of MARL benchmarks** (including Dispersion and Navigation from VMAS, SMAX, MAMuJoCo, and custom environments), diverse task types (homogeneous, heterogeneous, and mixed), and agent counts (2–20). We also show HyperMARL maintains *behavioural diversity* comparable to NoPS. |
| 🔌 **Easy integration** | Easy integration into existing on- or off-policy algorithms with minimal code changes — no extra loss terms, diversity hyperparameters (i.e. knowing the optimal diversity required for a task), or sequential updates. We include JAX (main) and Pytorch implementations. |


---

## Goals 🎯

|        | Description                 | Link                                    |
|--------|-----------------------------|------------------------------------------|
| 🚀     | **Quick Look**             | [`quickstart.ipynb`](quickstart.ipynb)                       |
| 📈     | **Reproduce results**       | [`docs/reproduce.md`](docs/reproduce.md) |
| 📚     | **Read the paper**          | [arXiv:2412.04233](https://arxiv.org/abs/2412.04233) |
| 🔍     | **Example (JAX Train + Non-JAX Env)**          | [`ippo_hypermarl`](baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py)                 |
| 🔍     | **Example (JAX Train + JAX Env)**          | [`ippo_rec_hypermarl`](baselines/IPPO/ippo_rnn_smax_mlp_hypernets_eval.py)                 |

---

## TODO
- [ ] Update HARL fork link.
- [x] Add quickstart notebook.

## Citation 📝
If you use HyperMARL in your work, please cite us as follows:
```bibtex
@article{tessera2024hypermarl,
  title={HyperMARL: Adaptive Hypernetworks for Multi-Agent RL},
  author={Tessera, Kale-ab Abebe and Rahman, Arrasy and Storkey, Amos and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2412.04233},
  year={2024}
}
```
