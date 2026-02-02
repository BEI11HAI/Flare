<div align="center">

  # FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning

  **IEEE Robotics and Automation Letters (RA-L), 2026**

  <div>
    <a href="https://scholar.google.com/citations?user=dXMBf_0AAAAJ&hl=zh-CN&oi=sra" target="_blank">Dongcheng Cao</a>,
    <a href="https://scholar.google.com/citations?user=ubQnCiQAAAAJ&hl=zh-CN&oi=sra" target="_blank">Jin Zhou</a>,
    <a href="https://scholar.google.com/citations?user=r8ugMOwAAAAJ&hl=zh-CN&oi=sra" target="_blank">Xian Wang</a>,
    <a href="https://scholar.google.com/citations?user=-MINoSEAAAAJ&hl=zh-CN&oi=sra" target="_blank">Shuo Li</a>
  </div>

  <small>
    College of Control Science and Engineering, Zhejiang University
  </small>

  <br><br>

  <a href="https://arxiv.org/abs/2508.09797"><img src="https://img.shields.io/badge/arXiv-2508.09797-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a> &nbsp;<a href="https://arxiv.org/pdf/2508.09797"><img src="https://img.shields.io/badge/Paper-PDF-EC1C24.svg?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="Paper PDF"></a> &nbsp; <a href="https://bei11hai.github.io/Flare-web/"><img src="https://img.shields.io/badge/Project-Website-blue?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Website"></a> &nbsp; <a href="https://youtu.be/CASn9SbnMHo"><img src="https://img.shields.io/badge/Video-YouTube-FF0000.svg?style=flat-square&logo=youtube&logoColor=white" alt="YouTube Video"></a> &nbsp; <a href="https://www.bilibili.com/video/BV1cBFTz6Eq7/?spm_id_from=333.1387.favlist.content.click&vd_source=3a757a9cdd97a3eeaf5f80ae50b97b4d"><img src="https://img.shields.io/badge/Video-Bilibili-FB7299.svg?style=flat-square&logo=bilibili&logoColor=white" alt="Bilibili Video"></a> &nbsp; <a href="https://github.com/BEI11HAI/Flare"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>

  <br><br>
  
  <p align="center">
    <img src="image/methodology.png" width="90%" alt="FLARE Methodology"/>
  </p>
</div>

---

## Abstract

**FLARE** is a reinforcement learning (RL) framework designed to tackle the formidable challenge of agile flight for quadrotor cable-suspended payload systems. Due to the underactuated, highly nonlinear, and hybrid dynamics of such systems, traditional methods often struggle.

In this work, we present a method that:
- **Directly learns** an agile navigation policy from high-fidelity simulation.
- Outperforms state-of-the-art optimization-based approaches (Impactor) by a **3x speedup** in gate traversal.
- Achieves successful **zero-shot sim-to-real transfer**, demonstrating remarkable agility and safety in real-world experiments.

## Installation

This repository depends on [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv) and the official [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator.

⚠️ **Important**: To ensure stable reproducibility, we **strongly recommend** using the specific commit version of Genesis that this project was developed on.

```bash
# 1. Clone Genesis and checkout the specific commit
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
git checkout 382cf4ca12c0c142adcf2fa7675eef65caf0c661
pip install -e .

# 2. Install rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
pip install -e .

# 3. Clone Flare
git clone https://github.com/BEI11HAI/Flare.git
cd Flare
```


## Usage

We provide three challenging scenarios for validation: **Agile Waypoint Passing**, **Payload Targeting**, and **Agile Gate Traversal**.

Below are the instructions to evaluate pre-trained models or train your own policies for **Scenario I: Agile Waypoint Passing**. (Instructions for other scenarios will be updated soon).

### 1. Evaluate Pre-trained Policy
You can directly evaluate the pre-trained model provided in `logs/s1_waypoint_passing`.

```bash
python waypoint_passing_eval.py
# Add --record to save a video
# python waypoint_passing_eval.py --record
```

Then you will see:

<p align="center">
  <img src="image/demo.gif" width="80%" alt="Demo GIF"/>
</p>

### 2. Train from Scratch
To train a new policy for the waypoint passing task:

```bash
python waypoint_passing_train.py
# python waypoint_passing_train.py --record 
```

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{cao2025flare,
  title={FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning},
  author={Cao, Dongcheng and Zhou, Jin and Wang, Xian and Li, Shuo},
  journal={arXiv preprint arXiv:2508.09797},
  year={2025}
}
```

---
<div align="center">
  Developed at <a href="http://nesc.zju.edu.cn/" target="_blank">NeSC-Lab</a>, Zhejiang University.
</div>
