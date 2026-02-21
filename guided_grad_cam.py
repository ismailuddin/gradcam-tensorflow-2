import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
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
    )
    import cv2

    return ResNet50, cv2, load_img, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guided Grad-CAM

    The Grad-CAM output can be improved further by combining with guided backpropagation, which zeroes elements in the gradients which act negatively towards the decision. Implementation from Raphael Meudec / [Sicara](https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow), [GitHub Gist](https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54).

    This output, however, is still a low resolution heatmap, and not quite as described in the original paper. The original paper
    """)
    return


@app.cell
def _(ResNet50, load_img, np, tf):
    image = np.array(load_img("./data/cat.jpg", target_size=(224, 224, 3)))
    model = ResNet50()

    # Get logits instead of softmax
    last_conv_layer = model.get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["avg_pool", "predictions"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return classifier_model, image, last_conv_layer_model, model


@app.cell
def _(classifier_model, image, last_conv_layer_model, np, tf):
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image[np.newaxis, ...])
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
    last_conv_layer_output = last_conv_layer_output[0]
    return grads, last_conv_layer_output


@app.cell
def _(cv2, grads, last_conv_layer_output, np, tf):
    guided_grads = (
        tf.cast(last_conv_layer_output > 0, "float32")
        * tf.cast(grads > 0, "float32")
        * grads
    )

    pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
    guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)

    for i, w in enumerate(pooled_guided_grads):
        guided_gradcam += w * last_conv_layer_output[:, :, i]

    guided_gradcam = cv2.resize(guided_gradcam.numpy(), (224, 224))

    guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
        guided_gradcam.max() - guided_gradcam.min()
    )
    return (guided_gradcam,)


@app.cell
def _(guided_gradcam, image, plt):
    plt.imshow(image)
    plt.imshow(guided_gradcam, alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guided Grad-CAM (high resolution maps)
    This approach reflects the paper's description better by first using the guided backpropagation approach to produce a high resolution map that is of the same resolution of the input image, which is then masked using the Grad-CAM heatmap to focus only on details that led to the prediction outcome. Based on the implementation on GitHub by [jacobgil](https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py).
    """)
    return


@app.cell
def _(tf):
    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

        return tf.nn.relu(x), grad

    return (guided_relu,)


@app.cell
def _(guided_relu, np, tf):
    class GuidedBackprop:
        def __init__(self, model, layer_name: str):
            self.model = model
            self.layer_name = layer_name
            self.gb_model = self.build_guided_model()

        def build_guided_model(self):
            gb_model = tf.keras.Model(
                self.model.inputs, self.model.get_layer(self.layer_name).output
            )
            layers = [
                layer for layer in gb_model.layers[1:] if hasattr(layer, "activation")
            ]
            for layer in layers:
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = guided_relu
            return gb_model

        def guided_backprop(self, image: np.ndarray):
            with tf.GradientTape() as tape:
                inputs = tf.cast(image, tf.float32)
                tape.watch(inputs)
                outputs = self.gb_model(inputs)
            grads = tape.gradient(outputs, inputs)[0]
            return grads

    return (GuidedBackprop,)


@app.cell
def _(GuidedBackprop, model):
    gb = GuidedBackprop(model, "conv5_block3_out")
    return (gb,)


@app.cell
def _(gb, guided_gradcam, image, np, tf):
    saliency_map = gb.guided_backprop(image[np.newaxis, ...]).numpy()
    saliency_map = saliency_map * np.repeat(guided_gradcam[..., np.newaxis], 3, axis=2)

    saliency_map -= saliency_map.mean()
    saliency_map /= saliency_map.std() + tf.keras.backend.epsilon()
    saliency_map *= 0.25
    saliency_map += 0.5
    saliency_map = np.clip(saliency_map, 0, 1)
    saliency_map *= (2**8) - 1
    saliency_map = saliency_map.astype(np.uint8)
    return (saliency_map,)


@app.cell
def _(plt, saliency_map):
    plt.imshow(saliency_map)
    return


if __name__ == "__main__":
    app.run()
