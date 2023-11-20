from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io
import base64
import torch
# from torchvision import transforms
# import torch.nn as nn
# from torch.autograd import Variable
import os
from perlin_numpy import generate_perlin_noise_2d
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def segment_image(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    return mask

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    img_data = data['imageData'].split(',')[1]
    file_bytes = io.BytesIO(base64.b64decode(img_data))
    image = cv2.imdecode(np.frombuffer(file_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fat_thresholds = [80, 150]
    muscle_thresholds = [1, 79]
    bone_thresholds = [151, 254]
    tissue_thresholds = [0, 0]

    components = {
        'fat': fat_thresholds,
        'muscle': muscle_thresholds,
        'bone': bone_thresholds,
        'tissue': tissue_thresholds
    }

    muscle_folder_path = 'static/muscle_images/muscle_images'
    muscle_images = [img for img in os.listdir(muscle_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    bone_folder_path = 'static/muscle_images/bone_images'
    bone_images = [img for img in os.listdir(bone_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fat_folder_path = 'static/muscle_images/fat_images'
    fat_images = [img for img in os.listdir(fat_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tissue_folder_path = 'static/muscle_images/tissue_images'
    tissue_images = [img for img in os.listdir(tissue_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    transformed = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    tissue_mask = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for component, (threshold_low, threshold_high) in components.items():
        mask = segment_image(gray, threshold_low, threshold_high)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(binary, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  
                cv2.drawContours(dilation, [cnt], -1, (0,0,0), -1)
        cleaned = cv2.dilate(dilation, kernel, iterations = 1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            x, y, w, h = bbox
            cv2.rectangle(cleaned, (x, y), (x+w, y+h), (127, 127, 127), 2)
            mask = np.zeros((h, w), dtype='uint8')
            shifted_contour = contour - np.array([x, y])
            cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
            if component == "boaxne":
                bone_canvas = np.zeros((h, w), dtype=np.uint8)
                perlin_noise = generate_perlin_noise_2d((256, 256), (16, 16))
                perlin_noise = cv2.normalize(perlin_noise, None, 0, 200, cv2.NORM_MINMAX)
                perlin_noise = np.uint8(perlin_noise)
                gradient = np.linspace(40, 40, h).astype(np.uint8)
                gradient_map = np.repeat(gradient, w).reshape(h, w)
                perlin_noise = cv2.resize(perlin_noise, (w,h))
                rippled_effect = cv2.addWeighted(perlin_noise, 0.4, gradient_map, 0.6, 0)
                combined_image = cv2.addWeighted(rippled_effect, 0.5, bone_canvas, 0.5, 0)
                combined_image = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)
                canvas[y:y+h, x:x+w][mask == 255] = combined_image[mask == 255]
            elif component == "bone":
                bone_image = cv2.imread(os.path.join(bone_folder_path, random.choice(bone_images)))
                bone_image = cv2.resize(bone_image, (w,h))
                canvas[y:y+h, x:x+w][mask == 255] = bone_image[mask == 255]
            elif component == "fat":
                fat_image = cv2.imread(os.path.join(fat_folder_path, random.choice(fat_images)))
                fat_image = cv2.resize(fat_image, (w,h))
                canvas[y:y+h, x:x+w][mask == 255] = fat_image[mask == 255]
            elif component == "muscle":
                muscle_image = cv2.imread(os.path.join(muscle_folder_path, random.choice(muscle_images)))
                muscle_image = cv2.resize(muscle_image, (w,h))
                canvas[y:y+h, x:x+w][mask == 255] = muscle_image[mask == 255]
            else:
                tissue_image = cv2.imread(os.path.join(tissue_folder_path, random.choice(tissue_images)))
                tissue_image = cv2.resize(tissue_image, (w,h))
                # blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
                canvas[y:y+h, x:x+w][mask == 255] = tissue_image[mask == 255]
                tissue_mask[y:y+h, x:x+w][mask == 255] = 255
        transformed += canvas
    if len(transformed.shape) == 3:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(transformed, 1, 255, cv2.THRESH_BINARY_INV)
    # dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    # gradient_fill = np.uint8(170 * (1 - dist_transform_normalized))
    # gradient_fill_inverted = cv2.subtract(255, gradient_fill)
    # blurred_random_gradient = cv2.GaussianBlur(gradient_fill_inverted, (21, 21), 0)
    # gradient_fill = np.where(mask == 255, blurred_random_gradient, 0)
    # result = np.where(mask == 255, 200, transformed)
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if len(tissue_mask.shape) == 3:
        tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2GRAY)
    dist_transform = cv2.distanceTransform(tissue_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    dist_transform = (dist_transform * 255).astype(np.uint8)
    blurred_mask = cv2.GaussianBlur(dist_transform, (15, 15), 0)
    blurred_mask[blurred_mask>0] = 255
    scaling_factor = 1.25
    blur = cv2.blur(transformed,(5,5),0)
    blur = cv2.multiply(blur, np.array([scaling_factor]))
    blur = np.clip(blur, 0, 255).astype(np.uint8)
    out = transformed.copy()
    out[blurred_mask>0] = blur[blurred_mask>0]
    # feathered = (transformed * blurred_mask / 255).astype(np.uint8)

    _, img_encoded = cv2.imencode('.jpg', out)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
