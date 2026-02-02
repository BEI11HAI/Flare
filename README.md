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

  <a href="https://arxiv.org/abs/2508.09797">
    <img src="https://img.shields.io/badge/arXiv-2508.09797-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2508.09797">
    <img src="https://img.shields.io/badge/Paper-PDF-EC1C24.svg?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="Paper PDF">
  </a>
  &nbsp;
  <a href="https://bei11hai.github.io/Flare-web/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Website">
  </a>
  &nbsp;
  <a href="https://youtu.be/CASn9SbnMHo">
    <img src="https://img.shields.io/badge/Video-YouTube-FF0000.svg?style=flat-square&logo=youtube&logoColor=white" alt="YouTube Video">
  </a>
  &nbsp;
  <a href="YOUR_BILIBILI_VIDEO_URL">
    <img src="https://img.shields.io/badge/Video-Bilibili-FB7299.svg?style=flat-square&logo=bilibili&logoColor=white" alt="Bilibili Video">
  </a>
  &nbsp;
  <a href="https://github.com/BEI11HAI/Flare">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg?style=flat-square&logo=github&logoColor=white" alt="GitHub">
  </a>

  <br><br>
  
  <p align="center">
    <img src="Flare-web/public/methodology.png" width="90%" alt="FLARE Methodology"/>
  </p>
</div>

---

## Abstract

**FLARE** is a reinforcement learning (RL) framework designed to tackle the formidable challenge of agile flight for quadrotor cable-suspended payload systems. Due to the underactuated, highly nonlinear, and hybrid dynamics of such systems, traditional methods often struggle.

In this work, we present a method that:
- **Directly learns** an agile navigation policy from high-fidelity simulation.
- Outperforms state-of-the-art optimization-based approaches (Impactor) by a **3x speedup** in gate traversal.
- Achieves successful **zero-shot sim-to-real transfer**, demonstrating remarkable agility and safety in real-world experiments.

## Usage

> **Note**: The code release is currently under preparation. Please stay tuned for updates!

The repository will provide:
1.  **Environment Setup**: Instructions for installing dependencies and the Genesis simulator.
2.  **Training**: Scripts to train the RL policy from scratch.
3.  **Inference**: Tools to run the pre-trained policy in simulation and on real hardware.

```bash
# Coming soon...
git clone https://github.com/BEI11HAI/Flare.git
cd Flare
# python train.py --task=agile_navigation
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

## License

This project is released under the [MIT License](LICENSE).

---
<div align="center">
  Developed at <a href="http://www.cse.zju.edu.cn/english/" target="_blank">NESC-Lab</a>, Zhejiang University.
</div>
