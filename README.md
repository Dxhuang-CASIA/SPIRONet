# SPIRONet

Official PyTorch implementation of **SPIRONet: Spatial-Frequency Learning and
Topological Channel Interaction Network for Vessel Segmentation**.

> [Jun. 2026] Our paper has been accepted by *Biomedical Signal Processing and Control* 🎉.

[![Paper](https://img.shields.io/badge/arXiv-2406.19749-b31b1b.svg)](https://arxiv.org/abs/2406.19749)

## Repository Structure

```text
SPIRONet/
├── model/
│   ├── cross_attention.py   # Spatial-frequency cross-attention
│   ├── graph_module.py      # Topological channel interaction
│   └── network.py           # SPIRONet architecture
├── dataset.py               # Dataset loader and data augmentation
├── metric.py                # F1, sensitivity, IoU, and MCC metrics
├── train.py                 # Training script
└── test.py                  # Evaluation script
```

## Requirements

- Python 3.8+
- CUDA-capable GPU
- PyTorch
- NumPy
- SciPy
- OpenCV
- Pillow
- tqdm

Install the required Python packages:

```bash
pip install torch numpy scipy opencv-python pillow tqdm
```

## Dataset Preparation

Organize each dataset as follows:

```text
<dataset_root>/
└── <dataset_name>/
    ├── train/
    │   ├── img/
    │   └── label/
    └── test/
        ├── img/
        └── label/
```

Each image and its corresponding binary segmentation mask must have the same
filename. Supported dataset names in the provided scripts are `DCA1`, `XCAD`,
`CAXF`, `CADSA`, and `ARCADE`.

## Training

Set `base_path`, `dataset_name`, `image_size`, and `ckpt` in `train.py`, then
create the checkpoint directory and run:

```bash
python train.py
```

The script trains the model with three random seeds (`3407`, `42`, and `924`)
and saves one checkpoint per seed. Use an image size of `300` for DCA1 and
`512` for the other datasets.

## Evaluation

Set `base_path`, `dataset_name`, `image_size`, and `ckpt` in `test.py`, then
run:

```bash
python test.py
```

The evaluation script reports sensitivity, F1 score, IoU, and MCC. Set
`save_vis = True` and configure the output path in `test.py` to save predicted
segmentation masks.

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2026spironet,
  title = {SPIRONet: Spatial-Frequency Learning and Topological Channel Interaction Network for Vessel Segmentation},
  author = {Huang, De-Xing and others},
  journal = {Biomedical Signal Processing and Control},
  year = {2026}
}
```
