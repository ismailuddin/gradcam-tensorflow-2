import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.applications.resnet50 import (
        ResNet50,
        decode_predictions,
    )
    import cv2

    return ResNet50, cv2, decode_predictions, load_img, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Counterfactual explanation

    By negating the value of $\frac{\partial y^{c}}{\partial A^{k}}$, we can produce a map of regions that would lower the network's confidence in its prediction. This is useful when two competing objects are present in| the image. We can produce a "counterfactual" image with these regions masked out, which should give a higher confidence in the original prediction.
    """)
    return


@app.cell
def _(ResNet50, tf):
    model = ResNet50()

    # Get logits instead of softmax
    logits_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    last_conv_layer = model.get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["avg_pool", "predictions"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return classifier_model, last_conv_layer_model, logits_model


@app.cell
def _(load_img, np):
    multiobject_image = np.array(
        load_img("./data/cat_and_dog.jpg", target_size=(224, 224, 3))
    )
    return (multiobject_image,)


@app.cell
def _(classifier_model, last_conv_layer_model, multiobject_image, np, tf):
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(
            multiobject_image[np.newaxis, ...]
        )
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    return last_conv_layer_output, tape, top_class_channel


@app.cell
def _(last_conv_layer_output, tape, top_class_channel):
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    return (grads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The negative of the gradients is taken, to target the regions that do not contribute towards strengthen the network's predictions.
    """)
    return


@app.cell
def _(grads, last_conv_layer_output, tf):
    pooled_grads = tf.reduce_mean(-1 * grads, axis=(0, 1, 2))
    last_conv_layer_output_ = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output_[:, :, i] *= pooled_grads[i]
    return (last_conv_layer_output_,)


@app.cell
def _(cv2, last_conv_layer_output_, np):
    # Average over all the filters to get a single 2D array
    ctfcl_gradcam = np.mean(last_conv_layer_output_, axis=-1)
    # Normalise the values
    ctfcl_gradcam = np.clip(ctfcl_gradcam, 0, np.max(ctfcl_gradcam)) / np.max(
        ctfcl_gradcam
    )
    ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (224, 224))
    return (ctfcl_gradcam,)


@app.cell
def _(ctfcl_gradcam, multiobject_image, plt):
    plt.imshow(multiobject_image)
    plt.imshow(ctfcl_gradcam, alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can mask out the region identified by the counterfactual map, and re-run the predictions, where we should see a higher confidence in the outputs.
    """)
    return


@app.cell
def _(ctfcl_gradcam, cv2, multiobject_image):
    mask = cv2.resize(ctfcl_gradcam, (224, 224))
    mask[mask > 0.1] = 255
    mask[mask != 255] = 0
    mask = mask.astype(bool)
    ctfctl_image = multiobject_image.copy()
    ctfctl_image[mask] = (0, 0, 0)
    return (ctfctl_image,)


@app.cell
def _(ctfctl_image, plt):
    plt.imshow(ctfctl_image)
    return


@app.cell
def _(ctfctl_image, decode_predictions, logits_model, np, tf):
    decode_predictions(
        tf.nn.softmax(logits_model(ctfctl_image[np.newaxis, ...])).numpy()
    )
    return


if __name__ == "__main__":
    app.run()
