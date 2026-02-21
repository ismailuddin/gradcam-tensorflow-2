import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, np, plt


@app.cell
def _():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.applications.resnet50 import (
        ResNet50,
        preprocess_input,
        decode_predictions,
    )
    import cv2

    return ResNet50, cv2, decode_predictions, load_img, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Grad-CAM output is an activation map which localises the detected objected to a region in the image. It is of width $u$ and height $v$, for the class $c$.
    $$
    L^{c}_{\textrm{Grad-CAM}} \in \mathbb{R}^{u \times v}
    $$
    """)
    return


@app.cell
def _(load_img, np, plt):
    image = np.array(load_img("./data/cat.jpg", target_size=(224, 224, 3)))
    plt.imshow(image)
    return (image,)


@app.cell
def _(ResNet50, tf):
    model = ResNet50()

    # Get logits instead of softmax
    logits_model = tf.keras.Model(
        inputs=model.inputs, outputs=model.layers[-1].output
    )
    return logits_model, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We get the output of the last convolution layer. We then create a model that goes up to only that layer.
    """)
    return


@app.cell
def _(model, tf):
    last_conv_layer = model.get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    return last_conv_layer, last_conv_layer_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a model which then takes the output of the model above, and uses the remaining layers to get the final predictions.
    """)
    return


@app.cell
def _(last_conv_layer, model, tf):
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["avg_pool", "predictions"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return (classifier_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we get the output from the model up till the last convolution layer.
    We ask `tf` to watch this tensor output, as we want to calculate the gradients of the predictions of our target class wrt to the output of this model (last convolution layer model).
    """)
    return


@app.cell
def _(classifier_model, image, last_conv_layer_model, np, tf):
    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    return last_conv_layer_output, tape, top_class_channel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The partial derivative / gradient of the model output (logits / prior to softmax), $y^{c}$, with respect to the feature map (filter) activations of a specified convolution layer (the last convolution layer in this case) is:
    $$
    \frac{\partial y^{c}}{\partial A^{k}_{ij}}
    $$
    """)
    return


@app.cell
def _(last_conv_layer_output, tape, top_class_channel):
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    return (grads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gradients have a shape of `(u,v,Z)`, where `(u,v)` comes from the shape of the 2D convolution filter (i.e. width and height), and `Z` is the number of filters. The next step averages each of the filters to a single value, so that the final shape is `Z` or the number of filters. This is equivalent to the global average pooling 2D layer.

    $$
    \alpha_{k}^{c}=\frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^{c}}{\partial A^{k}_{ij}}
    $$

    Each one of these gradients represents the connection from one of the pixels in the 2D array to the neuron / output representing the target class
    """)
    return


@app.cell
def _(grads, tf):
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    return (pooled_grads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is what the next layer in the model does which is a global average pooling 2D layer, which  averages and flattens the $z$ number of filters of $u \times v$ shape to single numbers (exactly what we did in previous step). This is necessary to create a connection to the fully connected (Dense) layers for the final prediction outputs.

    The next step is to multiply the gradients (corresponding to the importance of the given feature map / filter) with the actual feature map (filter) it represents.

    $$
    ReLU\bigg(\sum_{k} a^{c}_{k}A^{k}\bigg)
    $$
    """)
    return


@app.cell
def _(last_conv_layer_output, pooled_grads):
    last_conv_layer_output_ = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.numpy().shape[-1]):
        last_conv_layer_output_[:, :, i] *= pooled_grads[i]
    return (last_conv_layer_output_,)


@app.cell
def _(cv2, last_conv_layer_output_, np):
    # Average over all the filters to get a single 2D array
    grad_cam = np.mean(last_conv_layer_output_, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    grad_cam = np.clip(grad_cam, 0, np.max(grad_cam)) / np.max(grad_cam)
    grad_cam = cv2.resize(grad_cam, (224, 224))
    return (grad_cam,)


@app.cell
def _(grad_cam, image, plt):
    plt.imshow(image)
    plt.imshow(grad_cam, alpha=0.5)
    return


@app.cell
def _(decode_predictions, image, logits_model, np, tf):
    decode_predictions(tf.nn.softmax(logits_model(image[np.newaxis, ...])).numpy())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
