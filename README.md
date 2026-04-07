# EyesOnYou: Smart Exam Proctoring System

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8_Nano-00FFFF.svg?style=flat&logo=yolo&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)
![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat)

A lightweight, CPU-friendly computer vision application designed to monitor remote examinations and ensure academic integrity. Built entirely in Python, this system uses state-of-the-art object detection and facial landmark tracking to detect potential cheating behaviors in real-time without requiring expensive cloud GPUs.

## Key Features

* **Real-Time Monitoring:** Processes live webcam feeds at smooth frame rates directly on standard client hardware (CPUs).
* **Identity & Presence Tracking:** Uses Google MediaPipe to ensure the candidate remains in the frame and alerts if multiple faces are detected.
* **Gaze & Head Pose Estimation:** Calculates facial landmark ratios (Eye/Nose/Cheek geometry) to detect if the candidate is frequently looking away at off-screen materials.
* **Unauthorized Object Detection:** Integrates YOLOv8 Nano to instantly recognize cell phones or physical books entering the frame.
* **Smart Debouncing & Memory:** Utilizes bounding-box memory and event debouncing to prevent UI flickering and alert spam.
* **Automated Incident Reporting:** Generates a timestamped `.csv` log of all violations and saves corresponding photographic evidence (JPEGs) locally.

## Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Computer Vision:** [OpenCV](https://opencv.org/)
* **Facial Tracking:** [Google MediaPipe](https://developers.google.com/mediapipe)
* **Object Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Nano Model)
* **Data Logging:** Pandas & Python CSV

## System Architecture

The project is strictly modular, separating the Streamlit UI from the heavy AI processing logic.

    eyes-on-you/
    ├── app.py                  # Main Streamlit UI and application loop
    ├── core/                   # AI Processing Modules
    │   ├── face_tracker.py     # MediaPipe face detection
    │   ├── pose_estimator.py   # MediaPipe facial landmark & gaze math
    │   └── object_detector.py  # YOLOv8 object detection
    ├── utils/                  
    │   └── logger.py           # Handles CSV logging and evidence snapshots
    ├── data/                   # Generated locally during runtime
    │   ├── logs/               # Session CSV reports
    │   └── snapshots/          # Photographic evidence of violations
    ├── models/                 # Auto-downloaded YOLOv8 weights (.pt)
    ├── requirements.txt        
    └── README.md               

## Installation & Setup

Because this system relies heavily on AI libraries, it is highly recommended to run this inside a virtual environment using **Python 3.11**.

**1. Clone the repository:**
``` bash
git clone https://github.com/YOUR_USERNAME/eyes-on-you.git
cd eyes-on-you
```

**2. Create and activate a virtual environment:**
    # Example using pyenv or standard venv
``` bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install PyTorch (CPU Version):**
*Note: To save hundreds of megabytes and ensure smooth running on laptops, install the CPU-only version of PyTorch first.*
``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. Install remaining dependencies:**
``` bash
pip install -r requirements.txt
```

## Usage

To start the proctoring dashboard, run:
``` bash
streamlit run app.py
```
1. Check **"Start Proctoring Session"** in the sidebar to activate the webcam and AI monitoring.
2. The system will alert you on-screen of any violations.
3. Uncheck the session box to end the exam.
4. Check **"View Session Report"** to view the auto-generated incident log.
5. Check the `/data/snapshots/` folder for photo evidence.

## Known Limitations & Engineering Trade-offs

* **COCO Dataset Limitations:** The current YOLOv8n model is trained on the standard COCO dataset. It is highly accurate at detecting phone screens and camera lenses but may struggle to identify the featureless backs of modern smartphones. V2 would require a custom-trained dataset.
* **Optical Flow:** To maintain high FPS on CPUs, object detection is run periodically rather than continuously. Bounding boxes are held in memory between AI scans, which may cause minor visual lag if an object moves extremely fast across the screen.

## License
This project is licensed under [MIT License](LICENSE).