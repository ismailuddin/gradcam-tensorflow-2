# Grad-CAM TensorFlow 2 Implementation

This repository provides an implementation of **Grad-CAM (Gradient-weighted Class Activation Mapping)** using **TensorFlow 2 and Keras**. Grad-CAM is a visualization technique that helps understand which regions of an image contribute the most to a model’s prediction.

Grad-CAM is widely used for **model interpretability in deep learning**, especially in computer vision tasks.

---

## What is Grad-CAM?

Grad-CAM highlights the important regions in an input image that influence a model's prediction by using the gradients flowing into the last convolutional layer.

It helps answer the question:

> *Why did the model make this prediction?*

Grad-CAM is commonly used for:

- Model interpretability  
- Debugging neural networks  
- Visualizing CNN attention  
- Trustworthy AI systems  

---

## Features

- TensorFlow 2 compatible implementation  
- Works with **pretrained CNN models (ResNet50, etc.)**  
- Generates **class activation heatmaps**  
- Visualizes model attention on input images  

---

## Requirements

Python version used:

```
Python 3.10
```

Main dependencies:

```
tensorflow >= 2.10
numpy
matplotlib
opencv-python
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the Grad-CAM notebook:

```
notebooks/GradCam.ipynb
```

The notebook demonstrates:

1. Loading a pretrained model (ResNet50)  
2. Running predictions on an image  
3. Generating Grad-CAM heatmaps  
4. Visualizing the model attention  

---

## Important Note (Logits vs Softmax)

Grad-CAM gradients should be computed using **logits rather than post-softmax probabilities**.

However, `decode_predictions()` in Keras expects probability values.

To ensure correct behavior, softmax is applied before decoding predictions:

```python
preds = tf.nn.softmax(logits_model(image[np.newaxis, ...]))
decode_predictions(preds.numpy())
```

This ensures compatibility with Keras utilities while preserving correct Grad-CAM gradient computation.

---

## Example Output

The output of Grad-CAM is a **heatmap overlay** showing the regions of the image that influenced the prediction.

---

## Contribution

Contributions are welcome!  
If you find bugs or improvements, feel free to submit a pull request.

---

## Reference

Selvaraju et al., 2017  

**Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**

Paper:  
https://arxiv.org/abs/1610.02391

---

## Author

TensorFlow Grad-CAM implementation adapted for TensorFlow 2.
