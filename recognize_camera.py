#!/usr/bin/env python3
"""
Convenience script for camera-based face recognition
"""

import sys
import argparse
from face_recognition_yolo import YOLOFaceRecognizer


def main():
    parser = argparse.ArgumentParser(
        description='Real-time face recognition from camera using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recognize_camera.py
  python recognize_camera.py --camera-id 1
  python recognize_camera.py --db my_faces.pkl --confidence 0.6
  
Controls:
  Press 'q' to quit
        """
    )
    
    parser.add_argument('--camera-id', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--db', default='face_database.pkl', 
                       help='Face database path (default: face_database.pkl)')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    print("Initializing face recognition system...")
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,
        yolo_model=args.model,
        confidence=args.confidence
    )
    
    # Start camera recognition
    print(f"\nStarting camera {args.camera_id}...")
    recognizer.process_camera(camera_id=args.camera_id)


if __name__ == '__main__':
    main()