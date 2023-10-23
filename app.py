from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    img_data = data['imageData'].split(',')[1]
    file_bytes = io.BytesIO(base64.b64decode(img_data))
    image = cv2.imdecode(np.fromstring(file_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Map colors to grayscale values (You may need to adjust thresholds)
    _, muscle = cv2.threshold(gray, 85, 100, cv2.THRESH_BINARY)  # Muscle Color
    _, fat = cv2.threshold(gray, 150, 162, cv2.THRESH_BINARY)    # Fat Color
    _, bone = cv2.threshold(gray, 195, 205, cv2.THRESH_BINARY)   # Bone Color

    # Combine them
    ultrasound_image = np.maximum(muscle, fat)
    ultrasound_image = np.maximum(ultrasound_image, bone)

    # Add speckle noise
    noise = np.random.normal(0, 25, ultrasound_image.shape).astype(np.uint8)
    ultrasound_image = cv2.add(ultrasound_image, noise)

    # Apply Gaussian blur
    ultrasound_image = cv2.GaussianBlur(ultrasound_image, (5,5), 0)

    _, img_encoded = cv2.imencode('.jpg', ultrasound_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
