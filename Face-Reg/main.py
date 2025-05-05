"""
main.py - Main Entry Point for Face Recognition System

This is the main entry point that provides command-line access to the different
modules of the face recognition system.
"""

import argparse
import sys
import os

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='Face Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py enroll              # Run staff enrollment
  python main.py recognize           # Run face recognition
  python main.py recognize --camera 1 --location "Store Entrance"
        '''
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Enrollment parser
    enroll_parser = subparsers.add_parser('enroll', help='Run staff enrollment')
    
    # Recognition parser
    recognize_parser = subparsers.add_parser('recognize', help='Run face recognition')
    recognize_parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID to use (default: 0)')
    recognize_parser.add_argument('--location', type=str, default="Main Entrance",
                        help='Location identifier for this camera (default: Main Entrance)')
    recognize_parser.add_argument('--detector', type=str, default="mtcnn",
                        help='Face detector to use: mtcnn or hog (default: mtcnn)')
    recognize_parser.add_argument('--model', type=str, default="facenet",
                        help='Face recognition model: facenet or mobilefacenet (default: facenet)')
    
    # Analytics parser
    analytics_parser = subparsers.add_parser('analytics', help='Generate analytics reports')
    analytics_parser.add_argument('--days', type=int, default=30,
                        help='Number of days to include in reports (default: 30)')
    analytics_parser.add_argument('--output', type=str, default="reports",
                        help='Output directory for reports (default: reports)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == 'enroll':
        print("Starting staff enrollment...")
        try:
            # Import here to avoid circular imports
            from staff_enrollment import run as run_enrollment
            run_enrollment()
        except ImportError:
            print("Error: Staff enrollment module not found")
            return 1
            
    elif args.command == 'recognize':
        print(f"Starting face recognition with camera {args.camera} at location '{args.location}'...")
        print(f"Using detector: {args.detector}, model: {args.model}")
        try:
            # Import here to avoid circular imports
            from recognition_system import run as run_recognition
            run_recognition(
                camera_id=args.camera,
                location=args.location,
                detector_type=args.detector,
                embedding_model=args.model
            )
        except ImportError:
            print("Error: Recognition system module not found")
            return 1
            
    elif args.command == 'analytics':
        print(f"Generating analytics reports for the past {args.days} days...")
        try:
            # Import here to avoid circular imports
            import analytics
            analytics.generate_reports(days=args.days, output_dir=args.output)
        except ImportError:
            print("Error: Analytics module not found")
            return 1
            
    else:
        parser.print_help()
        return 0
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 