#!/usr/bin/env python3
"""
Convenience script for image-based face recognition
"""

import sys
import argparse
from face_recognition_yolo import YOLOFaceRecognizer


def main():
    parser = argparse.ArgumentParser(
        description='Recognize faces in an image using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recognize_image.py photo.jpg
  python recognize_image.py photo.jpg --output result.jpg
  python recognize_image.py photo.jpg --db my_faces.pkl --confidence 0.6
        """
    )
    
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output', '-o', help='Path to save output image')
    parser.add_argument('--db', default='face_database.pkl', help='Face database path (default: face_database.pkl)')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no-show', action='store_true', help='Do not display result window')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    print("Initializing face recognition system...")
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,
        yolo_model=args.model,
        confidence=args.confidence
    )
    
    # Process image
    print(f"\nProcessing image: {args.image}")
    recognizer.process_image(
        args.image,
        output_path=args.output,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()