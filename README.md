# 🔭 Grad-CAM implemenation in TensorFlow 2.X
[![DOI](https://zenodo.org/badge/286249315.svg)](https://zenodo.org/badge/latestdoi/286249315)
>   A TensorFlow 2.X implementation of the various uses of Grad-CAM the original paper, including counterfactual examples and guided Grad-CAM.

![Example](./data/example.jpg)

## Usage
Three separate Marimo notebooks are provided for:

- GradCam
- Counter-factual GradCam
- Guided GradCam with high saliency maps

They can be viewed using `uv run marimo edit`.

## Requirements
- [uv](https://docs.astral.sh/uv/)

## References
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
- https://github.com/jacobgil/pytorch-grad-cam
