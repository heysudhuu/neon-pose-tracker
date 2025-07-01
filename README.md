# My Neon Pose Tracker

## Overview
My Neon Pose Tracker is an interactive application that utilizes computer vision and machine learning to track human poses and hand landmarks in real-time. Built with PyQt5 for the GUI, OpenCV for video capture, and MediaPipe for pose detection, this application allows users to visualize their movements in a 3D space while providing functionalities such as video recording, GIF exporting, and background music playback.

## Features
- Real-time pose and hand tracking using MediaPipe
- 3D visualization of tracked coordinates
- Video recording functionality
- Export recorded sessions as GIFs
- Background music playback

## Installation
To set up the project, ensure you have Python installed on your machine. Then, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/my-neon-pose-tracker.git
   cd my-neon-pose-tracker
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/neon_pose_tracker_gui.py
   ```

2. Use the GUI to start recording, export GIFs, and play background music.

3. Adjust the camera and ensure proper lighting for optimal pose detection.

## Dependencies
The project requires the following Python libraries:
- OpenCV
- MediaPipe
- PyQt5
- imageio
- pygame

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.