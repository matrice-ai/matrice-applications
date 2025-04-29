# People Queue Counter

A tool to count people in multiple queues using YOLOv8 object detection and custom tracking.

## Features

- People detection using YOLOv8
- Support for multiple queue areas with distinct colors
- Buffer time for queue counting (only count people who stay in the queue)
- Unique person counting per queue (total unique visitors to each queue)
- Custom object tracking with trajectory visualization
- Interactive queue area definition
- Queue count trend analysis over time

## Setup

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your video file named `cr.mp4` in the same directory as the main.py file

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Define queue areas:
   - You'll be asked how many queue areas you want to define
   - A grid will be displayed over the first frame of the video
   - For each queue area, you can specify:
     - A custom name (e.g., "Checkout Queue", "Information Desk")
     - 3-4 points to define the polygon shape
   - Each queue area will be assigned a unique color automatically

3. Video processing:
   - The script will process the video without displaying intermediate frames
   - Progress information will be printed in the console
   - People in each queue area will be counted only if they stay for at least 10 seconds
   - Each queue will track unique visitors (people who entered the queue at any point)

4. Results:
   - Processed video will be saved as `queue_output.mp4`
   - A queue trend graph will be displayed and saved as `queue_trends.png`
   - Final statistics will show unique counts for each queue

## How It Works

1. **Object Detection**: YOLOv8 detects people in each video frame
2. **Object Tracking**: A custom IoU-based tracker assigns IDs to detected people
3. **Queue Counting with Buffer Time**: People are only counted if they remain in the queue area for a specified duration
4. **Unique Counting**: Each person who enters a queue is counted exactly once in the unique total
5. **Multiple Queue Areas**: Each defined area is tracked separately with its own statistics
6. **Visualization**: Bounding boxes, IDs, tracks, and queue counts are visualized with appropriate colors

## Customization

You can modify the following parameters in main.py:

- `model_path`: Path to a custom YOLOv8 model
- `confidence`: Detection confidence threshold (default: 0.4)
- `buffer_time_seconds`: How long a person must be in a queue to be counted (default: 10 seconds)
- `display_every`: Display every Nth frame during processing (default: 0 = no display) 