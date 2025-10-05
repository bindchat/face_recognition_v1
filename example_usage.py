#!/usr/bin/env python3
"""
Example usage of the YOLO Face Recognition system
This demonstrates how to use the API programmatically
"""

from face_database import FaceDatabase
from face_recognition_yolo import YOLOFaceRecognizer


def example_database_management():
    """Example: Managing face database"""
    print("=" * 50)
    print("EXAMPLE 1: Database Management")
    print("=" * 50)
    
    # Create database instance
    db = FaceDatabase('example_database.pkl')
    
    # Add a face (you would replace with actual image path)
    # db.add_face_from_image('path/to/photo.jpg', 'Person Name')
    
    # Import from directory structure
    # db.import_from_directory('faces/')
    
    # List all faces
    db.list_faces()
    
    print("\n")


def example_image_recognition():
    """Example: Recognize faces in an image"""
    print("=" * 50)
    print("EXAMPLE 2: Image Recognition")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = YOLOFaceRecognizer(
        db_path='face_database.pkl',
        yolo_model='yolov8n.pt',
        confidence=0.5
    )
    
    # Process an image
    # results = recognizer.process_image(
    #     'test_image.jpg',
    #     output_path='result.jpg',
    #     show=True
    # )
    
    # Results format: [(name, confidence, bbox), ...]
    # for name, conf, (x1, y1, x2, y2) in results:
    #     print(f"Found: {name} with confidence {conf:.2f}")
    
    print("Initialize a recognizer and call process_image() with your image path")
    print("\n")


def example_camera_recognition():
    """Example: Real-time camera recognition"""
    print("=" * 50)
    print("EXAMPLE 3: Camera Recognition")
    print("=" * 50)
    
    # Initialize recognizer
    # recognizer = YOLOFaceRecognizer(
    #     db_path='face_database.pkl',
    #     yolo_model='yolov8n.pt',
    #     confidence=0.5
    # )
    
    # Start camera recognition
    # recognizer.process_camera(camera_id=0)
    
    print("Initialize a recognizer and call process_camera() to start")
    print("Press 'q' to quit the camera view")
    print("\n")


def main():
    print("\n")
    print("╔" + "=" * 60 + "╗")
    print("║" + " " * 10 + "YOLO FACE RECOGNITION EXAMPLES" + " " * 19 + "║")
    print("╚" + "=" * 60 + "╝")
    print("\n")
    
    example_database_management()
    example_image_recognition()
    example_camera_recognition()
    
    print("=" * 50)
    print("QUICK START GUIDE")
    print("=" * 50)
    print("""
1. Organize face images in directory structure:
   faces/
   ├── person1/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   └── person2/
       └── photo1.jpg

2. Import faces into database:
   $ python face_database.py import faces/

3. Recognize faces in image:
   $ python recognize_image.py photo.jpg

4. Recognize faces from camera:
   $ python recognize_camera.py

For more information, see README.md
    """)


if __name__ == '__main__':
    main()