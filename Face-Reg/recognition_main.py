"""
recognition_main.py - Entry point for the Face Recognition System

This script launches the real-time face recognition system which can:
- Detect and recognize staff members from enrollment database
- Track and identify returning customers
- Monitor unknown faces and enroll frequent visitors
"""

import argparse
from face_recognition_module import run

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID to use (default: 0)')
    parser.add_argument('--location', type=str, default="Main Entrance",
                        help='Location identifier for this camera (default: Main Entrance)')
    
    args = parser.parse_args()
    
    print(f"Starting Face Recognition System")
    print(f"Camera ID: {args.camera}")
    print(f"Location: {args.location}")
    
    # Run the face recognition system
    run(camera_id=args.camera, location=args.location) 