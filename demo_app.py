from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from PIL import Image
import subprocess
import os
import base64
import io
import json
import time

app = Flask(__name__)

class MedSAMFlaskUI:
    def __init__(self):
        # C++ Model paths
        self.encoder_path = "openvino_models/rep_medsam_preprocessed/encoder.xml"
        self.decoder_path = "openvino_models/rep_medsam_preprocessed/decoder.xml"
        self.cache_dir = "cpp/data/cache"
        self.cpp_executable = "cpp/build/main"
        
        # Python script path
        self.python_script = "infer_rep_medsam.py"
        
        # Store current image and boxes
        self.current_image = None
        self.boxes = []
        
        print("MedSAM Flask UI initialized - ready for dual inference")
    
    def run_python_inference(self, image, boxes):
        """Run Python MedSAM inference by calling infer_rep_medsam.py"""
        try:
            if not os.path.exists(self.python_script):
                return None, 0, f"Python script not found: {self.python_script}"
            
            # Create input and output directories for Python inference
            python_input_dir = "input_python"
            python_output_dir = "output_python"
            os.makedirs(python_input_dir, exist_ok=True)
            os.makedirs(python_output_dir, exist_ok=True)
            
            # Prepare input data
            input_file = os.path.join(python_input_dir, "2D_input.npz")
            output_file = os.path.join(python_output_dir, "2D_input.npz")
            
            # Convert boxes to numpy array
            boxes_array = np.array(boxes, dtype=np.float32)
            
            # Save as NPZ file
            np.savez_compressed(input_file, imgs=image, boxes=boxes_array)
            
            # Run Python script
            cmd = [
                "python", self.python_script,
                "-i", python_input_dir,
                "-o", python_output_dir,
                "-device", "cpu"  # Use CPU to avoid GPU compatibility issues
            ]
            
            start_time = time.time()

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if result.returncode != 0:
                return None, inference_time, f"Python inference failed: {result.stderr}"
            
            # Load results
            if os.path.exists(output_file):
                result_data = np.load(output_file)
                segmentation_result = result_data['segs']
                return segmentation_result, inference_time, "Success"
            else:
                return None, inference_time, "No output file generated"
                
        except Exception as e:
            return None, 0, f"Python inference error: {str(e)}"
    
    def run_cpp_inference(self, image, boxes):
        """Run C++ MedSAM inference"""
        try:
            
            if not os.path.exists(self.cpp_executable):
                return None, 0, f"C++ executable not found: {self.cpp_executable}"
            
            # Create input and output directories
            input_dir = "input"
            output_dir = "output"
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare input data
            input_file = os.path.join(input_dir, "2D_input.npz")
            output_file = os.path.join(output_dir, "2D_input.npz")
            
            # Convert boxes to numpy array
            boxes_array = np.array(boxes, dtype=np.float32)
            
            # Save as NPZ file
            np.savez_compressed(input_file, imgs=image, boxes=boxes_array)
            
            # Run C++ executable
            cmd = [
                self.cpp_executable, self.encoder_path, self.decoder_path,
                self.cache_dir, input_dir, output_dir
            ]
            
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if result.returncode != 0:
                return None, inference_time, f"C++ inference failed: {result.stderr}"
            
            # Load results
            if os.path.exists(output_file):
                result_data = np.load(output_file)
                segmentation_result = result_data['segs']
                return segmentation_result, inference_time, "Success"
            else:
                return None, inference_time, "No output file generated"
                
        except Exception as e:
            return None, 0, f"C++ inference error: {str(e)}"

    def process_image_and_boxes(self, image_data, boxes):
        """Process image and run both segmentation methods"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.current_image = np.array(image)
            self.boxes = boxes
            
            if len(boxes) == 0:
                return {"success": False, "message": "Please draw at least one bounding box"}
            
            # Run both inference methods
            cpp_segs, cpp_time, cpp_status = self.run_cpp_inference(self.current_image, boxes)
            python_segs, python_time, python_status = self.run_python_inference(self.current_image, boxes)
            # python_segs, python_time, python_status = None, 0, "_"

            # Create colored segmentation for both results
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
            ]
            
            results = {}
            
            # Process C++ results
            if cpp_segs is not None:
                cpp_colored_seg = np.zeros((*cpp_segs.shape, 3), dtype=np.uint8)
                for i in range(1, len(boxes) + 1):
                    mask = cpp_segs == i
                    color = colors[(i - 1) % len(colors)]
                    cpp_colored_seg[mask] = color
                
                alpha = 0.5
                cpp_overlay = cv2.addWeighted(self.current_image.astype(np.uint8), 1 - alpha, 
                                            cpp_colored_seg.astype(np.uint8), alpha, 0)
                
                cpp_overlay_pil = Image.fromarray(cpp_overlay)
                buffer = io.BytesIO()
                cpp_overlay_pil.save(buffer, format='PNG')
                cpp_overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                results["cpp_result"] = f"data:image/png;base64,{cpp_overlay_b64}"
                results["cpp_time"] = f"{cpp_time:.3f}s"
                results["cpp_status"] = cpp_status
            else:
                results["cpp_result"] = None
                results["cpp_time"] = f"{cpp_time:.3f}s"
                results["cpp_status"] = cpp_status
            
            # Process Python results
            if python_segs is not None:
                python_colored_seg = np.zeros((*python_segs.shape, 3), dtype=np.uint8)
                for i in range(1, len(boxes) + 1):
                    mask = python_segs == i
                    color = colors[(i - 1) % len(colors)]
                    python_colored_seg[mask] = color
                
                alpha = 0.5
                python_overlay = cv2.addWeighted(self.current_image.astype(np.uint8), 1 - alpha, 
                                               python_colored_seg.astype(np.uint8), alpha, 0)
                
                python_overlay_pil = Image.fromarray(python_overlay)
                buffer = io.BytesIO()
                python_overlay_pil.save(buffer, format='PNG')
                python_overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                results["python_result"] = f"data:image/png;base64,{python_overlay_b64}"
                results["python_time"] = f"{python_time:.3f}s"
                results["python_status"] = python_status
            else:
                results["python_result"] = None
                results["python_time"] = f"{python_time:.3f}s"
                results["python_status"] = python_status
            
            # Determine overall success
            success = (cpp_segs is not None) or (python_segs is not None)
            
            if success:
                message = f"Inference completed!\n"
                message += f"Rep-MedSAM C++: {results['cpp_time']} ({results['cpp_status']})\n"
                message += f"Rep-MedSAM Python: {results['python_time']} ({results['python_status']})"
            else:
                message = f"Both inference methods failed:\n"
                message += f"C++: {cpp_status}\n"
                message += f"Python: {python_status}"
            
            return {"success": success, "message": message, "results": results}
                
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

# Initialize the UI
medsam_ui = MedSAMFlaskUI()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>MedSAM Inference Comparison</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5; 
        }
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            color: #333;
        }
        .main-content { 
            display: flex; 
            gap: 30px; 
        }
        .image-section { 
            flex: 2; 
        }
        .controls-section { 
            flex: 1; 
            background: #f8f9fa; 
            padding: 25px; 
            border-radius: 8px; 
            height: fit-content;
        }
        #canvas { 
            border: 3px solid #ddd; 
            cursor: crosshair; 
            max-width: 100%; 
            border-radius: 8px;
        }
        .upload-area { 
            border: 3px dashed #007bff; 
            padding: 30px; 
            text-align: center; 
            margin-bottom: 20px; 
            border-radius: 8px; 
            background: #f8f9ff;
        }
        .upload-area:hover { 
            background: #e6f3ff; 
        }
        button { 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 6px; 
            cursor: pointer; 
            margin: 8px 4px; 
            font-size: 14px;
            transition: background 0.3s;
        }
        button:hover { 
            background: #0056b3; 
        }
        button:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
        }
        .clear-btn { 
            background: #dc3545; 
        }
        .clear-btn:hover { 
            background: #c82333; 
        }
        .run-btn { 
            background: #28a745; 
            font-size: 16px;
            padding: 15px 30px;
        }
        .run-btn:hover { 
            background: #218838; 
        }
        #status { 
            margin: 15px 0; 
            padding: 12px; 
            border-radius: 6px; 
            font-weight: bold;
            white-space: pre-line;
        }
        .success { 
            background: #d4edda; 
            color: #155724; 
            border: 1px solid #c3e6cb;
        }
        .error { 
            background: #f8d7da; 
            color: #721c24; 
            border: 1px solid #f5c6cb;
        }
        .info { 
            background: #d1ecf1; 
            color: #0c5460; 
            border: 1px solid #bee5eb;
        }
        .boxes-list { 
            background: white; 
            padding: 15px; 
            border-radius: 6px; 
            margin: 15px 0;
            border: 1px solid #ddd;
        }
        .box-item { 
            padding: 8px; 
            margin: 5px 0; 
            background: #f8f9fa; 
            border-radius: 4px;
            font-family: monospace;
        }
        .instructions { 
            background: #fff3cd; 
            padding: 20px; 
            border-radius: 6px; 
            margin-bottom: 20px;
            border: 1px solid #ffeaa7;
        }
        .instructions h3 { 
            margin-top: 0; 
            color: #856404;
        }
        .instructions ol { 
            margin: 10px 0; 
            padding-left: 20px;
        }
        .instructions li { 
            margin: 8px 0; 
        }
        .results-section {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .result-item {
            flex: 1;
            text-align: center;
        }
        .result-item h3 {
            margin-top: 0;
        }
        .result-item img {
            max-width: 100%;
            border-radius: 8px;
            border: 3px solid;
        }
        .cpp-result {
            border-color: #007bff !important;
        }
        .python-result {
            border-color: #28a745 !important;
        }
        .result-time {
            font-weight: bold;
            margin-top: 10px;
            padding: 8px;
            border-radius: 4px;
        }
        .cpp-time {
            color: #007bff;
            background: #e7f3ff;
        }
        .python-time {
            color: #28a745;
            background: #e8f5e8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MedSAM Inference Comparison</h1>
            <p>Compare Rep-MedSAM C++ vs Python implementations with timing</p>
        </div>
        
        <div class="main-content">
            <div class="image-section">
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <h3>📁 Click to Upload Image</h3>
                    <p>Supports: PNG, JPG, JPEG, BMP, TIFF</p>
                    <input type="file" id="imageInput" accept="image/*" style="display: none;">
                </div>
                
                <canvas id="canvas" style="display: none;"></canvas>
                
                <!-- Results section -->
                <div id="resultsSection" style="display: none;">
                    <div class="results-section">
                        <div class="result-item">
                            <h3 style="color: #007bff;">Rep-MedSAM C++</h3>
                            <img id="cppResultImage" class="cpp-result">
                            <div id="cppTime" class="result-time cpp-time"></div>
                        </div>
                        <div class="result-item">
                            <h3 style="color: #28a745;">Rep-MedSAM Python</h3>
                            <img id="pythonResultImage" class="python-result">
                            <div id="pythonTime" class="result-time python-time"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="controls-section">
                <div class="instructions">
                    <h3>📋 Instructions</h3>
                    <ol>
                        <li>Upload a medical image</li>
                        <li>Click and drag to draw rectangles</li>
                        <li>Draw multiple boxes for different regions</li>
                        <li>Click "Run Inference" to compare both methods</li>
                    </ol>
                </div>
                
                <button onclick="clearBoxes()" class="clear-btn">Clear All Boxes</button>
                
                <div class="boxes-list">
                    <h4>Current Bounding Boxes:</h4>
                    <div id="boxesList">No boxes drawn yet</div>
                </div>
                
                <button onclick="runSegmentation()" class="run-btn" id="runBtn" disabled>Run Inference</button>
                
                <div id="status" class="info">Ready - Upload an image to start</div>
            </div>
        </div>
    </div>

    <script>
        let canvas, ctx;
        let isDrawing = false;
        let startX, startY;
        let boxes = [];
        let currentImage = null;

        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    loadImage(event.target.result);
                };
                reader.readAsDataURL(file);
            }
        });

        function loadImage(src) {
            const img = new Image();
            img.onload = function() {
                canvas = document.getElementById('canvas');
                ctx = canvas.getContext('2d');
                
                // Set canvas size
                const maxWidth = 800;
                const maxHeight = 600;
                let width = img.width;
                let height = img.height;
                
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }
                if (height > maxHeight) {
                    width = (width * maxHeight) / height;
                    height = maxHeight;
                }
                
                canvas.width = width;
                canvas.height = height;
                canvas.style.display = 'block';
                
                // Draw image
                ctx.drawImage(img, 0, 0, width, height);
                currentImage = src;
                boxes = [];
                updateBoxesList();
                updateStatus('Image loaded! Draw bounding boxes by clicking and dragging', 'info');
                
                // Hide results section
                document.getElementById('resultsSection').style.display = 'none';
            };
            img.src = src;
        }

        // Mouse events for drawing
        document.addEventListener('DOMContentLoaded', function() {
            document.addEventListener('mousedown', startDrawing);
            document.addEventListener('mousemove', draw);
            document.addEventListener('mouseup', stopDrawing);
        });

        function startDrawing(e) {
            if (!canvas || e.target !== canvas) return;
            
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
        }

        function draw(e) {
            if (!isDrawing || !canvas || e.target !== canvas) return;
            
            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            // Redraw image and existing boxes
            redrawCanvas();
            
            // Draw current rectangle
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        }

        function stopDrawing(e) {
            if (!isDrawing || !canvas) return;
            
            isDrawing = false;
            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;
            
            // Only add box if it has meaningful size
            if (Math.abs(endX - startX) > 10 && Math.abs(endY - startY) > 10) {
                const box = [
                    Math.min(startX, endX),
                    Math.min(startY, endY),
                    Math.max(startX, endX),
                    Math.max(startY, endY)
                ];
                boxes.push(box);
                updateBoxesList();
                redrawCanvas();
                updateStatus(`Box added! Total boxes: ${boxes.length}`, 'success');
            }
        }

        function redrawCanvas() {
            if (!canvas || !currentImage) return;
            
            const img = new Image();
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw all boxes
                boxes.forEach((box, index) => {
                    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
                    ctx.strokeStyle = colors[index % colors.length];
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                    
                    // Draw box number
                    ctx.fillStyle = colors[index % colors.length];
                    ctx.font = '16px Arial';
                    ctx.fillText((index + 1).toString(), box[0] + 5, box[1] + 20);
                });
            };
            img.src = currentImage;
        }

        function updateBoxesList() {
            const boxesList = document.getElementById('boxesList');
            const runBtn = document.getElementById('runBtn');
            
            if (boxes.length === 0) {
                boxesList.innerHTML = 'No boxes drawn yet';
                runBtn.disabled = true;
            } else {
                boxesList.innerHTML = boxes.map((box, index) => 
                    `<div class="box-item">Box ${index + 1}: (${Math.round(box[0])}, ${Math.round(box[1])}) → (${Math.round(box[2])}, ${Math.round(box[3])})</div>`
                ).join('');
                runBtn.disabled = false;
            }
        }

        function clearBoxes() {
            boxes = [];
            updateBoxesList();
            redrawCanvas();
            updateStatus('All boxes cleared', 'info');
            
            // Show canvas and hide results
            if (canvas && currentImage) {
                canvas.style.display = 'block';
                document.getElementById('resultsSection').style.display = 'none';
            }
        }

        function runSegmentation() {
            if (!currentImage || boxes.length === 0) {
                updateStatus('Please upload an image and draw bounding boxes first', 'error');
                return;
            }
            
            updateStatus('Running dual inference... Please wait', 'info');
            document.getElementById('runBtn').disabled = true;
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: currentImage,
                    boxes: boxes
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('runBtn').disabled = false;
                
                if (data.success) {
                    updateStatus(data.message, 'success');
                    
                    // Hide canvas and show results
                    canvas.style.display = 'none';
                    document.getElementById('resultsSection').style.display = 'block';
                    
                    // Display C++ results
                    const cppImg = document.getElementById('cppResultImage');
                    const cppTime = document.getElementById('cppTime');
                    if (data.results.cpp_result) {
                        cppImg.src = data.results.cpp_result;
                        cppImg.style.display = 'block';
                        cppTime.textContent = `Time: ${data.results.cpp_time}`;
                    } else {
                        cppImg.style.display = 'none';
                        cppTime.textContent = `Failed: ${data.results.cpp_status}`;
                    }
                    
                    // Display Python results
                    const pythonImg = document.getElementById('pythonResultImage');
                    const pythonTime = document.getElementById('pythonTime');
                    if (data.results.python_result) {
                        pythonImg.src = data.results.python_result;
                        pythonImg.style.display = 'block';
                        pythonTime.textContent = `Time: ${data.results.python_time}`;
                    } else {
                        pythonImg.style.display = 'none';
                        pythonTime.textContent = `Failed: ${data.results.python_status}`;
                    }
                } else {
                    updateStatus(data.message, 'error');
                }
            })
            .catch(error => {
                document.getElementById('runBtn').disabled = false;
                updateStatus('Error: ' + error.message, 'error');
            });
        }

        function updateStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = type;
        }
    </script>
</body>
</html>
    ''')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    result = medsam_ui.process_image_and_boxes(data['image'], data['boxes'])
    return jsonify(result)

if __name__ == '__main__':
    print("Starting MedSAM Dual Inference Comparison...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
