import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'model.h5'

try:
    model = keras.models.load_model(MODEL_PATH)
    model_loaded = True
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")
    model_loaded = False
    model = None

MUDRA_CLASSES = {
    0: 'Palm',
    1: 'Fist',
    2: 'Thumbs Up',
    3: 'Peace',
    4: 'Okay'
}


def preprocess_image(image, target_size=(224, 224)):
    try:
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


def predict_gesture(image):
    if not model_loaded or model is None:
        return None, 'Model not loaded'

    try:
        prepared = preprocess_image(image)
        if prepared is None:
            return None, 'Image preprocessing failed'

        predictions = model.predict(prepared, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        gesture_name = MUDRA_CLASSES.get(predicted_index, f'Unknown ({predicted_index})')

        return {
            'gesture': gesture_name,
            'confidence': confidence,
            'class_index': predicted_index,
            'all_predictions': {str(i): float(predictions[0][i]) for i in range(len(predictions[0]))}
        }, None
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, str(e)


@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        uploaded = request.files['image']
        if uploaded.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500

        image_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result, error = predict_gesture(image)

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result), 200
    except Exception as e:
        print(f"Error in /api/predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/webcam')
def webcam_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Cannot open webcam')
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if model_loaded:
                result, _ = predict_gesture(frame)
                if result:
                    text = f"{result['gesture']} ({result['confidence']:.2f})"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'mudra_classes': MUDRA_CLASSES,
        'max_file_size_mb': 16
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded = request.files['file']
        if uploaded.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500

        filename = secure_filename(uploaded.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image format'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result, error = predict_gesture(image)

        os.remove(filepath)

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result), 200
    except Exception as e:
        print(f"Error in /api/upload: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print('Starting Mudra CNN Web Application...')
    print(f"Model Status: {'Loaded' if model_loaded else 'Not loaded'}")
    print('Visit http://localhost:5000 to access the application')
    app.run(debug=True, host='0.0.0.0', port=5000)
