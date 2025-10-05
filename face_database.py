#!/usr/bin/env python3
"""
Face Database Management Tool
Imports face images and creates face encodings database
"""

import os
import pickle
import argparse
import face_recognition
import cv2
from pathlib import Path


class FaceDatabase:
    def __init__(self, db_path='face_database.pkl'):
        self.db_path = db_path
        self.face_encodings = []
        self.face_names = []
        self.load_database()
    
    def load_database(self):
        """Load existing face database"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data.get('encodings', [])
                    self.face_names = data.get('names', [])
                print(f"✓ Loaded {len(self.face_names)} faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.face_encodings = []
                self.face_names = []
        else:
            print("No existing database found. Creating new database.")
    
    def save_database(self):
        """Save face database to file"""
        data = {
            'encodings': self.face_encodings,
            'names': self.face_names
        }
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Database saved with {len(self.face_names)} faces")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_face_from_image(self, image_path, name):
        """Add a face from an image file"""
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return False
        
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) == 0:
                print(f"✗ No face detected in {image_path}")
                return False
            
            if len(face_encodings) > 1:
                print(f"⚠ Multiple faces detected in {image_path}, using first face")
            
            # Add face encoding
            self.face_encodings.append(face_encodings[0])
            self.face_names.append(name)
            print(f"✓ Added face for '{name}' from {image_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            return False
    
    def import_from_directory(self, directory_path):
        """
        Import faces from a directory structure:
        directory/
            person1/
                photo1.jpg
                photo2.jpg
            person2/
                photo1.jpg
        """
        if not os.path.exists(directory_path):
            print(f"✗ Directory not found: {directory_path}")
            return
        
        directory = Path(directory_path)
        added_count = 0
        
        # Iterate through subdirectories (each subdirectory = person)
        for person_dir in directory.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                print(f"\nProcessing {person_name}...")
                
                # Process each image in the person's directory
                for image_file in person_dir.iterdir():
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        if self.add_face_from_image(str(image_file), person_name):
                            added_count += 1
        
        print(f"\n✓ Import complete! Added {added_count} faces")
        self.save_database()
    
    def list_faces(self):
        """List all faces in database"""
        if not self.face_names:
            print("Database is empty")
            return
        
        print(f"\nFaces in database ({len(self.face_names)} total):")
        from collections import Counter
        name_counts = Counter(self.face_names)
        for name, count in sorted(name_counts.items()):
            print(f"  - {name}: {count} encoding(s)")
    
    def clear_database(self):
        """Clear all faces from database"""
        self.face_encodings = []
        self.face_names = []
        self.save_database()
        print("✓ Database cleared")


def main():
    parser = argparse.ArgumentParser(description='Face Database Management Tool')
    parser.add_argument('--db', default='face_database.pkl', help='Database file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add face command
    add_parser = subparsers.add_parser('add', help='Add a face from image')
    add_parser.add_argument('image', help='Path to image file')
    add_parser.add_argument('name', help='Person name')
    
    # Import directory command
    import_parser = subparsers.add_parser('import', help='Import faces from directory')
    import_parser.add_argument('directory', help='Directory containing person subdirectories')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all faces in database')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db = FaceDatabase(args.db)
    
    if args.command == 'add':
        db.add_face_from_image(args.image, args.name)
        db.save_database()
    
    elif args.command == 'import':
        db.import_from_directory(args.directory)
    
    elif args.command == 'list':
        db.list_faces()
    
    elif args.command == 'clear':
        response = input("Are you sure you want to clear the database? (yes/no): ")
        if response.lower() == 'yes':
            db.clear_database()


if __name__ == '__main__':
    main()