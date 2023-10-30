from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io
import base64
import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

app = Flask(__name__)

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()
        self.init_size = 256 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2)) #latent dim = 100
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

@app.route('/')
def index():
    return render_template('index.html')

def im_converterX(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((1, 1, 1))
    image = image.clip(0, 1)
    return image

def segment_image(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    return mask

def apply_gan_and_create_masked_image(generator, image, mask, bbox):
    x, y, w, h = bbox
    Tensor = torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (256, 100))))
    gen_imgs = generator(z) 
    img = gen_imgs[0][1] #.cuda()
    transform = transforms.Compose([transforms.Resize((h, w))])
    img = transform(img)
    generated = (im_converterX(img)*255).astype(np.uint8)
    masked = cv2.bitwise_and(generated, generated, mask=mask[max(y-10, 0):min(y+h+10, image.shape[0]), max(x-10, 0):min(x+w+10, image.shape[1])])
    return masked

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    img_data = data['imageData'].split(',')[1]
    file_bytes = io.BytesIO(base64.b64decode(img_data))
    image = cv2.imdecode(np.fromstring(file_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fat_thresholds = [80, 150]
    muscle_thresholds = [85, 100]
    bone_thresholds = [151, 255]

    components = {
        'fat': fat_thresholds,
        'muscle': muscle_thresholds,
        'bone': bone_thresholds
    }

    canvas = np.zeros_like(image)
    generator = torch.load('static/models/all_phantom_GANgenerator_50epochs.h5', map_location=torch.device('cpu'))

    for component, (threshold_low, threshold_high) in components.items():
        mask = segment_image(gray, threshold_low, threshold_high)
        # _, img_encoded = cv2.imencode('.jpg', mask)
        # img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        # return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            masked_generated = apply_gan_and_create_masked_image(generator, image, mask, bbox)
            x, y, w, h = bbox
            canvas[max(y-10, 0):min(y+h+10, image.shape[0]), max(x-10, 0):min(x+w+10, image.shape[1])] = masked_generated

            _, img_encoded = cv2.imencode('.jpg', canvas)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})

    _, img_encoded = cv2.imencode('.jpg', canvas)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
