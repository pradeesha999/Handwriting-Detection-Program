from flask import Flask, request, render_template, send_file, jsonify, Response
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import easyocr
import threading
import time
import base64

app = Flask(__name__, static_folder='static')

def enhance_image(image):
    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

    # Upscale the image
    scale_factor = 2
    width = int(denoised_image.shape[1] * scale_factor)
    height = int(denoised_image.shape[0] * scale_factor)
    upscaled_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    block_size = 19  # Use an odd number greater than 1
    C_value = 3
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_value)

    return upscaled_image, thresh_image

def perform_ocr(image, progress_callback):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    for i, res in enumerate(result):
        time.sleep(0.1)  # Simulate processing time
        progress = int((i + 1) / len(result) * 100)
        progress_callback(progress)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    
    print("File uploaded:", file.filename)  # Debugging message
    
    # Read the image file
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    image = np.array(Image.open(in_memory_file))

    # Enhance the image
    upscaled_image, thresh_image = enhance_image(image)

    # Save the original image in-memory
    original_image_io = BytesIO()
    Image.fromarray(image).save(original_image_io, format='PNG')
    original_image_io.seek(0)
    original_image_base64 = base64.b64encode(original_image_io.getvalue()).decode('utf-8')

    # Save the enhanced image in-memory
    enhanced_image_io = BytesIO()
    Image.fromarray(thresh_image).save(enhanced_image_io, format='PNG')
    enhanced_image_io.seek(0)
    enhanced_image_base64 = base64.b64encode(enhanced_image_io.getvalue()).decode('utf-8')

    # Store the enhanced image in a global variable to be accessed by the OCR endpoint
    global enhanced_image
    enhanced_image = thresh_image

    return jsonify({
        "original_image": original_image_base64,
        "enhanced_image": enhanced_image_base64
    })

@app.route('/ocr-progress', methods=['GET'])
def ocr_progress():
    def generate():
        progress = 0
        while progress < 100:
            time.sleep(0.5)  # Simulate time delay
            progress += 10
            yield f"data: {progress}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/ocr', methods=['POST'])
def ocr_image():
    if enhanced_image is None:
        return jsonify({"error": "No enhanced image available"}), 400
    
    try:
        progress = [0]
        def progress_callback(val):
            progress[0] = val
        
        # Perform OCR on the enhanced image
        ocr_result = perform_ocr(enhanced_image, progress_callback)
        annotated_image = enhanced_image.copy()

        # Annotate the image with the detected text
        for (bbox, text, prob) in ocr_result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple([int(val) for val in top_left])
            bottom_right = tuple([int(val) for val in bottom_right])
            annotated_image = cv2.rectangle(annotated_image, top_left, bottom_right, (0, 0, 255), 2)
            annotated_image = cv2.putText(annotated_image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the annotated image in-memory
        annotated_image_io = BytesIO()
        Image.fromarray(annotated_image).save(annotated_image_io, format='PNG')
        annotated_image_io.seek(0)

        # Extract detected text for display
        detected_text = "\n".join([text for (_, text, _) in ocr_result])

        # Encode the annotated image as base64
        annotated_image_base64 = base64.b64encode(annotated_image_io.getvalue()).decode('utf-8')

        return jsonify({
            "text": detected_text,
            "annotated_image": annotated_image_base64
        })

    except Exception as e:
        print(f"Error during OCR process: {e}")
        return jsonify({"error": "Error occurred during OCR process"}), 500

if __name__ == '__main__':
    enhanced_image = None  # Global variable to store the enhanced image
    app.run(debug=True)
