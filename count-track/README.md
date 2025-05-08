# Object Detection, Counting and Tracking System

This project implements a modular system for detecting, counting, and tracking objects in videos. The system is designed with separate modules that can work independently and be combined for different use cases.

## Project Structure

```
.
├── detection/
│   └── detector.py
├── counting/
│   └── counter.py
├── tracking/
│   └── tracker.py
├── utils/
│   └── video_utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Features

- Object detection using YOLOv8
- Object counting with time-based statistics
- Object tracking with unique IDs
- Modular design for flexibility
- JSON output for frame-wise results
- Support for people and vehicle detection

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script with a video file:
```bash
python main.py --video path/to/video.mp4
```

## Output

The system generates:
1. Annotated video with bounding boxes and counts
2. JSON file with frame-wise detection results
3. Time-based counting statistics 