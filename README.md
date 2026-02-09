# Computer Vision Practice

![OpenCV Banner](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/Logo.png)

----

## 📖 Overview

A comprehensive repository containing **11 foundational OpenCV tutorials** and **13 real-world computer vision applications**, designed to take you from basic image processing to advanced AI-powered vision systems. This collection demonstrates the full spectrum of computer vision capabilities, enabling machines to perceive, interpret, and interact with visual data.
- ✨ [Features](#-features)
- 🏗️ [Repository Structure](#--repository-structure)
- 🎯 [OpenCV Tutorials (1-11)](#-opencv-tutorials-1-11)
- 🚀 [Computer Vision Applications (1-13)](#-computer-vision-applications-1-13)
- 🛠️ [Technologies Stack](#--technologies-stack)
- ⚙️ [Environment Setup](#--environment-setup)
- 🚀 [Getting Started](#-getting-started)
- 🔧 [Customization](#-customization)
- 📊 [Expected Outputs](#-expected-outputs)
- ©️ [Certification](#--certification)
- 🤝 [Contributing](#-contributing)
- 🙏 [Acknowledgments](#-acknowledgments)
- 📄 [License](#-license)

---
## ✨ Features

> **🧠 Foundational Learning**: 11 step-by-step OpenCV tutorials covering core concepts
---
> **🚀 Real-World Applications**: 13 production-ready computer vision projects
---
> **🛠️ Multi-Platform**: Local processing, cloud services (AWS), and edge computing
---
> **📊 Diverse Domains**: Medical imaging, gesture control, OCR, object tracking, and more
---
> **⚡ Performance Optimized**: Efficient implementations with various AI frameworks
---

---
## 🏗️ Repository Structure

```
📁 Computer-Vision-Practice/
│
├── # 🎬 OPENCV TUTORIALS (11 Foundation Files)
├── 1)_OpenCV_Image_Inp_Outp.py
├── 2)_OpenCV_Video_Inp_Outp.py
├── 3)_OpenCV_Webcam_Inp_Outp.py
├── 4)_OpenCV_Resize_and_Crop.py
├── 5)_OpenCV_ColorSpaces.py
├── 6)_OpenCV_Blurs.py
├── 7)_OpenCV_Global_Threshold.py
├── 8)_OpenCV_Adaptive_Threshold.py
├── 9)_OpenCV_Edge_Detection.py
├── 10)_OpenCV_Drawing.py
└── 11)_OpenCV_Contours.py
│
├── 📁 # 🗃️ OPENCV COURSE JUPYTER FILES (11 Conceptual Knowledge Notes)
│   ├── 1_Getting_Started_with_Images.ipynb
│   ├── 2_Image_Manipulations_Crop_Resize_Flip_and_Modify_Pixels.ipynb
│   ├── 3_Annotating_Images.ipynb
│   ├── 4_Image_Enhancement_Mathematical_Operations.ipynb
│   ├── 5_Writing_Video_using_OpenCV.ipynb
│   ├── 6_Image_Alignment.ipynb
│   ├── 7_Panorama.ipynb
│   ├── 8_HDR.ipynb
│   ├── 9_Object_Tracking.ipynb
│   ├── 10_TF_Object_Detection.ipynb
│   └── 11_OpenPose.ipynb
|
├── 📁 # 🚀 APPLICATIONAL PROJECTS (13 Real-World Applications)
│   ├── 1)_Color_Detection_of_Objects/
│   ├── 2)_Face_Anonymizer_Image_Video_Webcam/
│   ├── 3)_Text_Detection_OCR/
│   ├── 4)_Image_Classifier_Empty_or_Not_Parking_Lot/
│   ├── 5)_Feature_Extraction_with_Inference/
│   ├── 6)_Emotion_Recognition_with_Face_Mask/
│   ├── 7)_Sign_Language_Detection_for_N_Alphabets/
│   ├── 8)_Pneumonia_Classifier_XRayIMGs/
│   ├── 9)_YoloV11Nano_Object_Tracking/
│   ├── 10)_AWS_Rekognition_FullAccess_IAM/
│   ├── 11)_Parking_Spot_Counter/
│   ├── 12)_AWS_Lambda_and_API_Gateway/
│   └── 13)_Hand_Gesture_Volume_Control/
|
├── 📁 # 🖼️ DEMO IMAGES (Visual Results Gallery)
│   │
│   ├── # Representative Image (1 Image)
│   │   └── Logo.png   → Image for This Readme 
│   │
│   ├── # OpenCV Tutorial Demos (11 Images)
│   │   ├── OpenCV1.png    → Image Input/Output Demo
│   │   ├── OpenCV2.png    → Video Processing Demo
│   │   ├── OpenCV3.png    → Webcam Capture Demo
│   │   ├── OpenCV4.png    → Resize & Crop Demo
│   │   ├── OpenCV5.png    → Color Spaces Demo
│   │   ├── OpenCV6.png    → Blur Effects Demo
│   │   ├── OpenCV7.png    → Global Thresholding Demo
│   │   ├── OpenCV8.png    → Adaptive Thresholding Demo
│   │   ├── OpenCV9.png    → Edge Detection Demo
│   │   ├── OpenCV10.png   → Drawing Functions Demo
│   │   └── OpenCV11.png   → Contour Detection Demo
│   │
│   └── # Applicational Project Demos (20 Images)
│       ├── AppProj1.png      → Color Detection Results
│       ├── AppProj2.png      → Face Anonymizer Output
│       ├── AppProj3.png      → Text Detection & OCR
│       ├── AppProj4.png      → Parking Lot Classification
│       ├── AppProj5.png      → Feature Extraction
│       ├── AppProj6.1.png    → Emotion Recognition - Happy
│       ├── AppProj6.2.png    → Emotion Recognition - Sad
│       ├── AppProj6.3.png    → Emotion Recognition - Angry
│       ├── AppProj7.1.png    → Sign Language - Letter A
│       ├── AppProj7.2.png    → Sign Language - Letter B
│       ├── AppProj7.3.png    → Sign Language - Letter C
│       ├── AppProj8.1.png    → Pneumonia X-Ray - Normal
│       ├── AppProj8.2.png    → Pneumonia X-Ray - Positive
│       ├── AppProj8.3.png    → Pneumonia X-Ray - Heatmap
│       ├── AppProj9.png      → YOLO Object Tracking
│       ├── AppProj10.png     → AWS Rekognition Dashboard
│       ├── AppProj11.png     → Parking Spot Counter UI
│       ├── AppProj12.png     → AWS Lambda API Response
│       ├── AppProj13.1.png   → Hand Gesture - Volume Up
│       └── AppProj13.2.png   → Hand Gesture - Volume Down
│
├── 📁 # 📥 INPUTS (8 Sample Input Files)
│   ├── dragon.jpg              → Colorful dragon for tutorials
│   ├── sample_video.mp4        → Sample video for processing
│   ├── cow_salt_pepper.png     → Noisy image for denoising
│   ├── bear.jpg                → Image for segmentation
│   ├── handwritten_text.png    → Handwritten notes for OCR
│   ├── messi.jpg               → Portrait for edge detection
│   ├── whiteboard.png          → Whiteboard for drawing demo
│   └── birds.jpg              → Multiple objects for contours
│
├── 📁 # 📤 OUTPUTS (13 Generated Output Files)
│   ├── dragon_bgr.jpg                    → BGR color space
│   ├── dragon_rgb.jpg                    → RGB color space  
│   ├── dragon_gray.jpg                   → Grayscale conversion
│   ├── dragon_hsv.jpg                    → HSV color space
│   ├── cleaned_cow_salt_pepper.png       → Denoised image
│   ├── bear_segmented.jpg                → Segmented bear
│   ├── handwritten_text_extracted_global.png → Global threshold OCR
│   ├── handwritten_text_extracted_adaptive.png → Adaptive threshold OCR
│   ├── messi_edge.jpg                    → Canny edge detection
│   ├── messi_edge_dilated.jpg            → Dilated edges
│   ├── messi_edge_eroded.jpg             → Eroded edges
│   ├── drawing_on_whiteboard.png         → Annotations demo
│   └── contoured_birds.jpg              → Detected contours
│
├── .gitignore             → Git ignore configuration
├── environment.yml        → Conda environment specification
├── requirements.txt       → Python package dependencies
└── README.md             → This documentation file
```
---
## 🎯 OpenCV Tutorials (1-11)
### 1. Image Input/Output
Learn to read, display, and save images in various formats using OpenCV's core functions.

![Image I/O Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV1.png)

### 2. Video Input/Output
Process video files frame-by-frame with efficient streaming techniques.

![Video Processing Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV2.png)

### 3. Webcam Processing
Real-time webcam capture and processing for interactive applications.

![Webcam Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV3.png)

### 4. Image Manipulation
Resizing, cropping, and geometric transformations for image preprocessing.

![Image Manipulation Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV4.png)

### 5. Color Space Conversions
BGR, RGB, HSV, Grayscale conversions and their practical applications.

![Color Spaces Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV5.png)

### 6. Image Blurring Techniques
Gaussian, Median, and Bilateral filtering for noise reduction and smoothing.

![Blur Effects Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV6.png)

### 7. Global Thresholding
Binary and Otsu's thresholding methods for image segmentation.

![Global Thresholding Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV7.png)

### 8. Adaptive Thresholding
Local thresholding techniques for uneven lighting conditions.

![Adaptive Thresholding Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV8.png)

### 9. Edge Detection
Canny, Sobel, and Laplacian edge detection algorithms.

![Edge Detection Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV9.png)

### 10. Drawing Functions
Annotations, shapes, and text overlays on images and videos.

![Drawing Functions Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV10.png)

### 11. Contour Detection
Finding and analyzing object boundaries in images.

![Contour Detection Demo](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/OpenCV11.png)


---
## 🚀 Computer Vision Applications (1-13)
### 1. Color Detection 🎨
Real-time object detection based on color ranges with adjustable HSV sliders.

![Color Detection Results](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj1.png)

### 2. Face Anonymizer 🎭
Privacy-preserving face blurring/masking for images, videos, and live streams.

![Face Anonymizer Output](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj2.png)

### 3. Text Detection & OCR 📝
Multi-engine OCR (Tesseract, EasyOCR) with text localization and extraction.

![Text Detection & OCR](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj3.png)

### 4. Parking Lot Classifier 🅿️
Binary classification for parking space occupancy using custom CNN.

![Parking Lot Classification](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj4.png)

### 5. Feature Extraction 🔍
Deep feature extraction with pre-trained models for image retrieval.

![Feature Extraction](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj5.png)

### 6. Emotion Recognition 😷
Facial emotion classification with mask detection using MediaPipe.

| Happy Emotion | Sad Emotion | Surprised Emotion |
|---------------|-------------|---------------|
| ![Happy Emotion](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj6.1.png) | ![Sad Emotion](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj6.2.png) | ![Angry Emotion](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj6.3.png) |

### 7. Sign Language Detection 🤟
Real-time ASL alphabet recognition with custom dataset and CNN.

| Letter K | Letter R | Letter A |
|----------|----------|----------|
| ![Sign Language A](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj7.1.png) | ![Sign Language B](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj7.2.png) | ![Sign Language C](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj7.3.png) |

### 8. Pneumonia Classifier 🏥
Medical image analysis for pneumonia detection from chest X-rays.

| Site Header | Normal X-Ray | Positive Case |
|--------------|---------------|------------------|
| ![Normal X-Ray](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj8.1.png) | ![Positive Case](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj8.2.png) | ![Heatmap Analysis](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj8.3.png) |

### 9. YOLOv11 Object Tracking 🎯
Real-time object detection and tracking with Ultralytics YOLO.

![YOLO Object Tracking](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj9.png)

### 10. AWS Rekognition Integration ☁️
Cloud-based face analysis and comparison using AWS services.

![AWS Rekognition Dashboard](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj10.png)

### 11. Parking Spot Counter 🚗
Automated counting of available parking spaces with perspective correction.

![Parking Spot Counter UI](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj11.png)

### 12. Serverless CV API ⚡
AWS Lambda + API Gateway deployment for scalable computer vision.

![AWS Lambda API Response](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj12.png)

### 13. Gesture Volume Control 🔊
Hand gesture recognition for system volume control using MediaPipe.

| Volume Up Gesture | Volume Down Gesture |
|-------------------|---------------------|
| ![Volume Up](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj13.2.png) | ![Volume Down](https://raw.githubusercontent.com/Kratugautam99/Computer-Vision-Practice/refs/heads/main/Demo%20Images/AppProj13.1.png) |

---
## 🛠️ Technologies Stack

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4?style=for-the-badge&logo=google&logoColor=white)

**Complete Stack**: OpenCV, TensorFlow, PyTorch, AWS (Lambda, S3, Rekognition, IAM, API Gateway), Tesseract OCR, Easy OCR, Skimage OCR,  MediaPipe, Streamlit, Detectron2, Ultralytics YOLO v11, Pillow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

---
## ⚙️ Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate opencvenv

# Verify installation
python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')"
```

### Option 2: Using Pip/Virtualenv (Python Version = 3.11.13)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker (Advanced)

```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "1)_OpenCV_Image_Inp_Outp.py"]
```
---
## 🚀 Getting Started

### Running Tutorials

```bash
# Run any tutorial
python "1)_OpenCV_Image_Inp_Outp.py"
python "9)_OpenCV_Edge_Detection.py"
```

### Running Applications

```bash
# Navigate to application directory
cd "Applicational_Projects/1)_Color_Detection_of_Objects"

# Run the application
python main.py  # or specific script name
```

---
## 🔧 Customization

Each application is modular and can be easily customized:

```python
# Example: Modify color detection ranges
LOWER_HSV = [20, 50, 50]  # Adjust for different colors
UPPER_HSV = [40, 255, 255]

# Example: Change model paths
MODEL_PATH = "custom_model.pth"
CONFIG_PATH = "custom_config.yaml"
```
---
## 📊 Expected Outputs

Each tutorial generates corresponding output files in the `Outputs/` directory. The `Demo_Images/` folder contains screenshots of expected results for all tutorials and applications.

---
## ©️ Certification
The Official OpenCV Certification by OpenCV Company => http://courses.opencv.org/certificates/dd2b718cc65d4abe8080812a8ca6842e

---
## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
## 🙏 Acknowledgments

- **OpenCV Community** for the incredible computer vision library
- **AWS** for cloud infrastructure and AI services
- **Ultralytics** for YOLO implementations
- **Google Research** for MediaPipe and TensorFlow

---
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  
**⭐ Star this repo if you find it helpful!**

*"The eye sees only what the mind is prepared to comprehend."* - Henri Bergson

</div>

---
