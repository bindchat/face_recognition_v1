#!/usr/bin/env python3
"""
YOLO Face Recognition Module
Combines YOLO for face detection with face recognition for identification
"""

import cv2
import numpy as np
import face_recognition
import pickle
import os
from ultralytics import YOLO


class YOLOFaceRecognizer:
    def __init__(self, db_path='face_database.pkl', yolo_model='yolov8n.pt', confidence=0.5):
        """
        Initialize YOLO Face Recognizer
        
        Args:
            db_path: Path to face database pickle file
            yolo_model: YOLO model path (will download if not exists)
            confidence: Confidence threshold for YOLO detection
        """
        self.db_path = db_path
        self.confidence = confidence
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load face database
        self.load_database()
        
        # Initialize YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model)
        print("✓ YOLO model loaded")
    
    def load_database(self):
        """Load face database"""
        if not os.path.exists(self.db_path):
            print(f"⚠ Warning: Face database not found at {self.db_path}")
            print("  Run 'python face_database.py import <directory>' to create database")
            return False
        
        try:
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
            
            print(f"✓ Loaded {len(self.known_face_names)} faces from database")
            return True
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            return False
    
    def recognize_faces_in_frame(self, frame):
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            list of tuples: [(name, confidence, bbox), ...]
                where bbox is (x1, y1, x2, y2)
        """
        results = []
        
        # Run YOLO detection
        yolo_results = self.yolo_model(frame, verbose=False)
        
        # Process detections
        for result in yolo_results:
            boxes = result.boxes
            
            for box in boxes:
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Filter by confidence
                if conf < self.confidence:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Extract face region
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size == 0:
                    continue
                
                # Convert BGR to RGB for face_recognition
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(rgb_face)
                
                if len(face_encodings) == 0:
                    # No face encoding found, just mark as "Unknown"
                    results.append(("Unknown", conf, (x1, y1, x2, y2)))
                    continue
                
                face_encoding = face_encodings[0]
                
                # Compare with known faces
                name = "Unknown"
                best_match_confidence = 0.0
                
                if len(self.known_face_encodings) > 0:
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    
                    # Find best match
                    best_match_idx = np.argmin(face_distances)
                    best_distance = face_distances[best_match_idx]
                    
                    # Convert distance to confidence (0.6 is typical threshold)
                    match_confidence = 1 - best_distance
                    
                    # If distance is small enough, consider it a match
                    if best_distance < 0.6:
                        name = self.known_face_names[best_match_idx]
                        best_match_confidence = match_confidence
                
                results.append((name, best_match_confidence, (x1, y1, x2, y2)))
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: BGR image from OpenCV
            results: list of (name, confidence, bbox) tuples
            
        Returns:
            frame with annotations
        """
        for name, confidence, (x1, y1, x2, y2) in results:
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{name} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_image(self, image_path, output_path=None, show=True):
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            show: Whether to display result
            
        Returns:
            list of (name, confidence, bbox) tuples
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Error: Could not read image {image_path}")
            return []
        
        # Recognize faces
        results = self.recognize_faces_in_frame(frame)
        
        # Draw results
        output_frame = self.draw_results(frame.copy(), results)
        
        # Display result
        if show:
            cv2.imshow('Face Recognition', output_frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, output_frame)
            print(f"✓ Result saved to {output_path}")
        
        # Print results
        print(f"\nDetected {len(results)} face(s):")
        for name, conf, bbox in results:
            print(f"  - {name} (confidence: {conf:.2f})")
        
        return results
    
    def process_camera(self, camera_id=0):
        """
        Process camera feed in real-time
        
        Args:
            camera_id: Camera device ID (default 0)
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {camera_id}")
            return
        
        print("✓ Camera opened. Press 'q' to quit.")
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            # Process every frame (or skip frames for better performance)
            if frame_count % 1 == 0:  # Process every frame
                results = self.recognize_faces_in_frame(frame)
                frame = self.draw_results(frame, results)
            
            frame_count += 1
            
            # Display frame
            cv2.imshow('Face Recognition (Press q to quit)', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Face Recognition')
    parser.add_argument('--db', default='face_database.pkl', help='Face database path')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    
    subparsers = parser.add_subparsers(dest='mode', help='Recognition mode')
    
    # Image mode
    image_parser = subparsers.add_parser('image', help='Process image file')
    image_parser.add_argument('input', help='Input image path')
    image_parser.add_argument('--output', help='Output image path')
    image_parser.add_argument('--no-show', action='store_true', help='Do not display result')
    
    # Camera mode
    camera_parser = subparsers.add_parser('camera', help='Process camera feed')
    camera_parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Initialize recognizer
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,
        yolo_model=args.model,
        confidence=args.confidence
    )
    
    if args.mode == 'image':
        recognizer.process_image(
            args.input,
            output_path=args.output,
            show=not args.no_show
        )
    
    elif args.mode == 'camera':
        recognizer.process_camera(camera_id=args.camera_id)


if __name__ == '__main__':
    main()