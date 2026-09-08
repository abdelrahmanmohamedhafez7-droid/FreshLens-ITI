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
