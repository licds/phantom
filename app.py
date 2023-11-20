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

# class DCGAN_Generator(nn.Module):
#     def __init__(self):
#         super(DCGAN_Generator, self).__init__()
#         self.init_size = 256 // 4
#         self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2)) #latent dim = 100
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 1, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

@app.route('/')
def index():
    return render_template('index.html')

# def im_converterX(tensor):
#     image = tensor.cpu().clone().detach().numpy()
#     image = image.transpose(1,2,0)
#     image = image * np.array((1, 1, 1))
#     image = image.clip(0, 1)
#     return image

def segment_image(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    return mask

# def apply_gan_and_create_masked_image(generator, image, mask, bbox):
#     x, y, w, h = bbox
    # Tensor = torch.FloatTensor
    # z = Variable(Tensor(np.random.normal(0, 1, (256, 100))))
    # gen_imgs = generator(z) 
    # print(gen_imgs.shape)
    # img = gen_imgs[0] #.cuda()
    # transform = transforms.Compose([transforms.Resize((h, w))])
    # img = transform(img)
    # generated = (im_converterX(img)*255).astype(np.uint8)

    # img = img.resize((256, 256)).convert('L')
    # masked = cv2.bitwise_and(generated, generated, mask=mask[max(y, 0):min(y+h, image.shape[0]), max(x, 0):min(x+w, image.shape[1])])
    # return masked

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    img_data = data['imageData'].split(',')[1]
    file_bytes = io.BytesIO(base64.b64decode(img_data))
    # image = cv2.imdecode(np.fromstring(file_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.imdecode(np.frombuffer(file_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fat_thresholds = [80, 150]
    muscle_thresholds = [1, 79]
    bone_thresholds = [151, 254]

    components = {
        'fat': fat_thresholds,
        'muscle': muscle_thresholds,
        'bone': bone_thresholds
    }

    musscle_folder_path = 'static/muscle_images/2207_clinical_generated_2048'
    musscle_images = [img for img in os.listdir(musscle_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    bone_folder_path = 'static/muscle_images/2207_clinical_generated_2048'
    bone_images = [img for img in os.listdir(bone_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fat_folder_path = 'static/muscle_images/2207_clinical_generated_2048'
    fat_images = [img for img in os.listdir(fat_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tissue_folder_path = 'static/muscle_images/2207_clinical_generated_2048'
    tissue_images = [img for img in os.listdir(tissue_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    transformed = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
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
            if component == "bone":
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
            else:
                random_image = cv2.imread(os.path.join(folder_path, random.choice(images)))
                random_image = cv2.resize(random_image, (w,h))
                canvas[y:y+h, x:x+w][mask == 255] = random_image[mask == 255]
        transformed += canvas
    if len(transformed.shape) == 3:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(transformed, 1, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    gradient_fill = np.uint8(170 * (1 - dist_transform_normalized))
    gradient_fill_inverted = cv2.subtract(255, gradient_fill)
    blurred_random_gradient = cv2.GaussianBlur(gradient_fill_inverted, (21, 21), 0)
    gradient_fill = np.where(mask == 255, blurred_random_gradient, 0)
    result = np.where(mask == 255, gradient_fill, transformed)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    _, img_encoded = cv2.imencode('.jpg', result)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
