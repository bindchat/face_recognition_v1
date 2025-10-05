# YOLO Face Recognition System

A comprehensive face recognition system that uses YOLO for face detection and deep learning for face identification. Supports both image and real-time camera recognition with an easy-to-use face database management tool.

## Features

- 🎯 **YOLO-based face detection** - Fast and accurate face detection using YOLOv8
- 👤 **Face recognition** - Identify known faces from your database
- 📸 **Image recognition** - Process individual images
- 📹 **Camera recognition** - Real-time recognition from webcam/camera
- 🗄️ **Database management** - Easy-to-use tool for managing face databases
- 🎨 **Visual feedback** - Bounding boxes and labels with confidence scores

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing `dlib` might require additional system dependencies:
- **Ubuntu/Debian:** `sudo apt-get install cmake libopenblas-dev liblapack-dev`
- **macOS:** `brew install cmake`
- **Windows:** Consider using pre-built wheels from [here](https://github.com/jloh02/dlib/releases)

### 2. Download YOLO model (automatic on first run)

The YOLOv8 model will be automatically downloaded on first use.

## Quick Start

### Step 1: Create a Face Database

Organize your face images in the following directory structure:

```
faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg
```

Then import the faces into the database:

```bash
python face_database.py import faces/
```

### Step 2: Recognize Faces

**From an image:**
```bash
python recognize_image.py photo.jpg
```

**From camera (real-time):**
```bash
python recognize_camera.py
```

Press `q` to quit the camera view.

## Usage Guide

### Face Database Management

The `face_database.py` script manages your face database.

#### Import faces from a directory:
```bash
python face_database.py import <directory>
```

#### Add a single face:
```bash
python face_database.py add <image_path> <person_name>
```

Example:
```bash
python face_database.py add john_photo.jpg "John Doe"
```

#### List all faces in database:
```bash
python face_database.py list
```

#### Clear the database:
```bash
python face_database.py clear
```

#### Use a custom database file:
```bash
python face_database.py --db my_faces.pkl import faces/
```

### Image Recognition

Process a single image file with `recognize_image.py`:

#### Basic usage:
```bash
python recognize_image.py photo.jpg
```

#### Save output to file:
```bash
python recognize_image.py photo.jpg --output result.jpg
```

#### Don't show the result window:
```bash
python recognize_image.py photo.jpg --output result.jpg --no-show
```

#### Use custom database and confidence:
```bash
python recognize_image.py photo.jpg --db my_faces.pkl --confidence 0.6
```

### Camera Recognition

Real-time face recognition from camera with `recognize_camera.py`:

#### Basic usage (default camera):
```bash
python recognize_camera.py
```

#### Use a different camera:
```bash
python recognize_camera.py --camera-id 1
```

#### Use custom database and settings:
```bash
python recognize_camera.py --db my_faces.pkl --confidence 0.6
```

**Controls:**
- Press `q` to quit

### Advanced Usage

#### Using the Python API directly:

```python
from face_recognition_yolo import YOLOFaceRecognizer

# Initialize recognizer
recognizer = YOLOFaceRecognizer(
    db_path='face_database.pkl',
    yolo_model='yolov8n.pt',
    confidence=0.5
)

# Process an image
results = recognizer.process_image('photo.jpg', output_path='result.jpg')

# Start camera recognition
recognizer.process_camera(camera_id=0)
```

#### Manage face database programmatically:

```python
from face_database import FaceDatabase

# Create/load database
db = FaceDatabase('face_database.pkl')

# Add a face
db.add_face_from_image('photo.jpg', 'John Doe')

# Import from directory
db.import_from_directory('faces/')

# List faces
db.list_faces()

# Save database
db.save_database()
```

## Configuration Options

### YOLO Models

You can use different YOLO models for different speed/accuracy tradeoffs:

- `yolov8n.pt` - Nano (fastest, default)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

Example:
```bash
python recognize_camera.py --model yolov8m.pt
```

### Confidence Threshold

Adjust the detection confidence threshold (default: 0.5):

```bash
python recognize_image.py photo.jpg --confidence 0.7
```

Lower values = more detections (may include false positives)
Higher values = fewer detections (may miss some faces)

## How It Works

1. **Face Detection:** YOLO detects faces in the image/frame
2. **Face Encoding:** Each detected face is converted to a 128-dimensional encoding
3. **Face Matching:** The encoding is compared with known faces in the database
4. **Recognition:** If a match is found (distance < 0.6), the face is labeled with the person's name

## Troubleshooting

### No faces detected
- Ensure faces are clearly visible and not too small
- Try lowering the `--confidence` threshold
- Check image quality and lighting

### Poor recognition accuracy
- Add more photos of each person to the database (3-5 recommended)
- Use photos with different angles and lighting
- Ensure photos are clear and faces are front-facing

### Camera not opening
- Check camera permissions
- Try different `--camera-id` values (0, 1, 2, etc.)
- Ensure no other application is using the camera

### dlib installation issues
- On Ubuntu: `sudo apt-get install cmake libopenblas-dev liblapack-dev`
- On macOS: `brew install cmake`
- Consider using pre-built wheels for Windows

## Project Structure

```
.
├── face_database.py          # Face database management tool
├── face_recognition_yolo.py  # Main face recognition module
├── recognize_image.py        # Image recognition script
├── recognize_camera.py       # Camera recognition script
├── requirements.txt          # Python dependencies
├── face_database.pkl         # Face database (generated)
└── README.md                 # This file
```

## Requirements

- Python 3.7+
- OpenCV
- YOLOv8 (ultralytics)
- face_recognition
- dlib
- numpy
- pillow

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for face detection
- [face_recognition](https://github.com/ageitgey/face_recognition) for face encoding and matching
- [OpenCV](https://opencv.org/) for image processing