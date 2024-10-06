  # Vehicle Detection and Counting with YOLOv8 🚗🚚🚌

Vehicle Detection and Counting is a Python-based project designed to detect, track, and count vehicles in video footage using the YOLOv8 model and ByteTrack algorithm. This project generates annotated video outputs with bounding boxes, labels, and vehicle counts, which can be valuable for traffic analysis and surveillance applications.

## Motivation

The purpose of this project is to demonstrate the application of deep learning models in real-world scenarios, specifically in traffic monitoring and management. By using state-of-the-art object detection and tracking techniques, this project aims to provide an efficient and accurate method for vehicle detection and counting.

## Features

- **Vehicle Detection**: Identifies and labels vehicles such as cars, motorcycles, buses, and trucks in video frames using the YOLOv8 model.
  
- **Vehicle Tracking**: Tracks the detected vehicles across video frames using the ByteTrack algorithm.
  
- **Line Zone Counting**: Counts vehicles as they cross a specified line within the video.
  
- **Annotated Video Output**: Generates an output video with bounding boxes, labels, and vehicle count annotations.

## Getting Started

To run this application locally, follow these steps:

### Prerequisites

- Python 3.6+
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/vehicle-detection-and-counting.git
   cd vehicle-detection-and-counting
   ```
2. Install dependencies:

   ```bash
    pip install ultralytics
    pip install supervision
   ```

## Screenshots & Videos

![image](https://github.com/user-attachments/assets/83cc3e97-3ece-4886-9de4-4bda76a200ea)

https://github.com/user-attachments/assets/62255053-6b9e-4fb1-94b1-2c936b0acf30


## Requirements
- Python 3.6+

## How It Works

1. **Model Setup**
   - The YOLOv8 model is loaded and prepared for inference.
   - Specific classes of interest, such as cars and trucks, are selected for detection.

2. **Frame Processing**
   - Frames are extracted from the input video.
   - For each frame, the model predicts the locations and classes of vehicles, which are then tracked across frames using the ByteTrack algorithm.

3. **Vehicle Tracking and Counting**
   - As vehicles are tracked, they are counted when they cross a specified line in the frame. This helps in monitoring the flow of traffic.

4. **Annotated Video Output**
   - The processed frames are compiled into an output video, with annotations showing the detected vehicles, their tracking paths, and count indicators.


## How to Use

1.**Set up the environment**:Ensure the required packages are installed and set the paths for the source video and model.

2.**Run the script**: Execute the script to process the video and generate the output.

3.Set the source video path  
 ```bash
   SOURCE_VIDEO_PATH = "path/to/your/YOUR_VIDEO.mp4"
   # Run the vehicle detection and tracking
   # Refer to the code in the repository for the complete workflow
   ```

## Results

The output video will include:

- **Bounding Boxes**: Indicating detected vehicles.
- **Labels**: Showing the type of vehicle and confidence score.
- **Tracking Paths**: Visual lines representing the movement of each vehicle.
- **Vehicle Count**: Displayed on the screen as vehicles cross the defined line.


## Authors
- Your Name - D.Tejesh Kumar

## Contact
- For any questions or suggestions, please open an issue or contact the repository owner at dtejesh05k@gmail.com.
 
   
