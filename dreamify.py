import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image as keras_image
import webbrowser

file_path = ""
result_path = ""

def preprocess_image(image_path):
    img = keras_image.load_img(image_path)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def deep_dream(image_path, result_path):
    original_img = preprocess_image(image_path)
    model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

    layer_settings = {"mixed4": 1.0, "mixed5": 1.5, "mixed6": 2.0, "mixed7": 2.5}
    outputs_dict = {layer.name: layer.output for layer in [model.get_layer(name) for name in layer_settings.keys()]}
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)

    def compute_loss(input_image):
        features = feature_extractor(input_image)
        loss = tf.zeros(shape=())
        for name in features.keys():
            coeff = layer_settings[name]
            activation = features[name]
            scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
            loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
        return loss

    @tf.function
    def gradient_ascent_step(img, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = compute_loss(img)
        grads = tape.gradient(loss, img)
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
        img += learning_rate * grads
        return loss, img

    def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
        for _ in range(iterations):
            loss, img = gradient_ascent_step(img, learning_rate)
            if max_loss is not None and loss > max_loss:
                break
        return img

    step = 0.01
    iterations = 100
    max_loss = 15.0

    for layer_name in layer_settings.keys():
        img = tf.identity(original_img)
        img = gradient_ascent_loop(img, iterations=iterations, learning_rate=step, max_loss=max_loss)
        keras_image.save_img(result_path, deprocess_image(img.numpy()))
    
    root.destroy()

def open_file_dialog():
    global file_path
    global result_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result_path = file_path.split('.')[0] + '_dream.png'
        deep_dream(file_path, result_path)

# GUI setup
root = tk.Tk()
root.title("Deep Dream GUI")

# Button to open file dialog
btn_open_file = tk.Button(root, text="Open Image", command=open_file_dialog)
btn_open_file.pack(pady=20)

root.mainloop()

print("File Path is : ", file_path)
print("Result Path is : ", result_path)

file_name = os.path.basename(file_path)
result_name = os.path.basename(result_path)

# Copy files to the current working directory
shutil.copy(file_path, file_name)
shutil.copy(result_path, result_name)

# Generate HTML string with image paths
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Viewer</title>
    <style>
        body {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            height: 100vh;
            margin: 10;
        }}
        img {{
            max-width: 80%;
            max-height: 150vh;
        }}
    </style>
</head>
<body>
    <div>
        <img src="{file_name}" alt="Original Image">
    </div>
    <div>
        <img src="{result_name}" alt="Incepted Image">
    </div>
</body>
</html>
"""

# Save the HTML string to a file
with open("result_viewer.html", "w") as html_file:
    html_file.write(html_content)

# Open the generated HTML file in the default web browser
webbrowser.open("result_viewer.html")