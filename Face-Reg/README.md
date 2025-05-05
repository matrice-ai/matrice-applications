# Face Recognition System

A modular face recognition system for staff identification and customer tracking in retail environments.

## Overview

This system provides real-time face recognition capabilities with separate modules for:

- Staff enrollment and identification
- Customer tracking and recognition
- Analytics and reporting

The system is designed to work with standard webcams and can be extended to support multiple camera streams.

## Features

- **Staff Enrollment**: Register staff members with facial embeddings from multiple angles
- **Staff Recognition**: Identify known staff members in real-time video
- **Customer Tracking**: Anonymously track returning customers
- **Unknown Face Handling**: Enroll frequent visitors as customers automatically
- **Real-time Visualization**: Display recognition results with bounding boxes and labels
- **Analytics**: Generate reports on visitor traffic, customer behavior, and staff presence
- **State-of-the-art Models**: Uses MTCNN for face detection and deep learning models for face embeddings

## System Architecture

The system is built with a modular architecture:

- **Models**: Face detection and recognition models (MTCNN, FaceNet, MobileFaceNet)
- **Face Processor**: Image preprocessing, face alignment, and quality assessment
- **Staff Recognizer**: Staff identification against enrollment database
- **Customer Tracker**: Customer recognition and tracking of unknown faces
- **Database Manager**: Local JSON database operations
- **Utils**: Visualization, analytics, and helper functions
- **Recognition System**: Main orchestrator that integrates all components
- **Analytics**: Reporting and visualization generation

## Requirements

- Python 3.8+
- OpenCV
- face_recognition
- facenet_pytorch
- torch
- numpy
- matplotlib (for analytics)
- pandas (for analytics)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Staff Enrollment

To enroll new staff members:

```
python main.py enroll
```

Follow the on-screen instructions to:
1. Enter staff details (name, ID, position)
2. Capture facial data from multiple angles
3. Review and confirm enrollment

### Face Recognition

To run the face recognition system:

```
python main.py recognize
```

Optional parameters:
- `--camera`: Camera ID (default: 0)
- `--location`: Location identifier (default: "Main Entrance")
- `--detector`: Face detector type ("mtcnn" or "hog", default: "mtcnn")
- `--model`: Face recognition model ("facenet" or "mobilefacenet", default: "facenet")

Example:
```
python main.py recognize --camera 1 --location "Store Entrance"
```

### Analytics

To generate analytics reports:

```
python main.py analytics
```

Optional parameters:
- `--days`: Number of days to include in reports (default: 30)
- `--output`: Output directory for reports (default: "reports")

## Controls

When running the face recognition system:

- `q` - Quit
- `m` - Toggle display mode (all/staff/customers)
- `s` - Toggle staff display
- `c` - Toggle customer display
- `u` - Toggle unknown faces display
- `+` - Increase processing frequency
- `-` - Decrease processing frequency

## Configuration

Key configuration parameters can be found at the top of each module file.

## Data Storage

All data is stored locally in JSON files within the `face_recognition_data` directory:
- `staff_database.json`: Staff records and embeddings
- `customer_database.json`: Anonymous customer records
- `visit_logs.json`: Recognition event logs
- `staff_frames/`: Directory containing enrolled staff face images

## License

[MIT License](LICENSE)

## Acknowledgements

- face_recognition library by Adam Geitgey
- MTCNN implementation from facenet-pytorch
- OpenCV community 