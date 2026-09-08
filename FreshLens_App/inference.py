"""Preprocessing shared by the combined notebook and application."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def image_to_jpeg(image, image_size=160):
    image = ImageOps.pad(image.convert('RGB'), (image_size, image_size),
                         method=Image.Resampling.BILINEAR, color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def prepare_image(content, image_size=160):
    with Image.open(BytesIO(content)) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        content = image_to_jpeg(image, image_size)
    image = tf.io.decode_jpeg(content, channels=3)
    return tf.cast(image, tf.float32).numpy()[None, ...]


def predict_product(model, class_names, content, image_size=160):
    probabilities = model(prepare_image(content, image_size), training=False).numpy()[0]
    if len(probabilities) != len(class_names):
        raise ValueError('Class count does not match model output.')
    indices = np.argsort(probabilities)[::-1][:3]
    results = []
    for index in indices:
        result = {
            'product': class_names[index],
            'score': float(probabilities[index])
        }
        results.append(result)
    return results


from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def load_artifacts():
    labels = json.loads((ROOT / "models/class_names.json").read_text())
    config = json.loads((ROOT / "models/model_config.json").read_text())
    model = tf.keras.models.load_model(ROOT / "models/best_model.keras", compile=False)
    assert model.input_shape[1:] == (160, 160, 3)
    assert model.output_shape[-1] == len(labels) == config["num_classes"]
    assert config["resize_method"] == "pil_bilinear_white_pad_then_jpeg95"
    return model, labels, config


def read_image(source):
    with Image.open(source) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Please use an image smaller than 25 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")
