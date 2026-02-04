# 🔢 Intelligent Digit Recognition System (OCR)

<div align="center">

![Digit Recognition Demo](demo.gif)

**Advanced Computer Vision for Multi-Digit Handwritten Recognition**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

</div>

---

## 📖 About The Project

An advanced Computer Vision application that uses a **Convolutional Neural Network (CNN)** to detect and recognize multiple handwritten digits from images. Unlike basic MNIST classifiers, this system features a robust image processing pipeline capable of handling high-resolution photos, varying light conditions, and multiple digits in a single image.

### 🎯 What Makes This Different?

- 🖼️ **Real-World Ready**: Works with actual photos, not just clean datasets
- 🔍 **Multi-Digit Detection**: Automatically finds and classifies multiple digits
- 📐 **Smart Preprocessing**: Preserves aspect ratios and handles various lighting
- 🎨 **User-Friendly**: Clean dark-mode GUI for easy interaction

---

## 🚀 Key Features

### 🔢 Multi-Digit Detection
Unlike basic MNIST models, this system uses **contour detection** to find and classify multiple digits in a single image, making it practical for real-world applications.

### 📸 High-Resolution Support
Automatically scales large images down to optimal processing size (1000px width) for consistent filtering and faster performance without sacrificing accuracy.

### 📐 Aspect Ratio Preservation
Uses **square-canvas padding** to ensure digits are not stretched during resizing to 28x28 pixels, significantly improving classification accuracy.

### 🎛️ Robust Preprocessing Pipeline
- **Otsu's Adaptive Thresholding**: Works in varying light conditions
- **Morphological Operations**: Removes paper grain, shadows, and noise
- **Gaussian Blur**: Reduces high-frequency artifacts
- **Smart Contour Filtering**: Ignores smudges and irrelevant shapes

### 🖥️ Interactive GUI
A clean, dark-mode desktop interface built with **Tkinter** featuring:
- Drag-and-drop image upload
- Real-time bounding box visualization
- Confidence score display
- Result export functionality

---

## 🛠️ Technical Architecture

### 1. The Model (CNN)

The "brain" of the system is a Convolutional Neural Network trained on the MNIST dataset with custom enhancements.

**Architecture:**
```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (128) + ReLU + Dropout(0.5)
    ↓
Dense (10) + Softmax
```

**Key Components:**
- **Conv2D Layers**: Extract features like edges, curves, and patterns
- **MaxPooling**: Reduces spatial dimensions while retaining important features
- **Dropout**: Prevents overfitting during training
- **Softmax**: Outputs probability distribution across 10 digit classes (0-9)

### 2. The Vision Pipeline

Every detected digit undergoes a carefully designed preprocessing pipeline:

```
Original Image
    ↓
1. Downsampling (if > 1000px wide)
    ↓
2. Grayscale Conversion
    ↓
3. Gaussian Blur (5x5 kernel)
    ↓
4. Otsu's Binary Thresholding
    ↓
5. Morphological Opening (noise removal)
    ↓
6. Contour Detection & Filtering
    ↓
7. Bounding Box Extraction
    ↓
8. Square Canvas Padding
    ↓
9. Resize to 28x28
    ↓
10. Normalization (0-1 range)
    ↓
CNN Prediction
```

### 3. Optimization Strategies

| Challenge | Solution | Benefit |
|-----------|----------|---------|
| **High-Res Images** | Image downsampling to 1000px width | Prevents "giant" contours, speeds up processing |
| **Thin Digits (e.g., 1)** | Proportional square-canvas padding | Prevents horizontal stretching during 28x28 resize |
| **Paper Grain/Noise** | Morphological Opening | Cleans tiny specks and shadows from background |
| **Over-detection** | Size & confidence filtering | Ignores smudges and low-confidence predictions |
| **Varying Lighting** | Otsu's adaptive thresholding | Works across different lighting conditions |

---

## 📋 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Basic understanding of command line

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/digit-ocr-system.git
   cd digit-ocr-system
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install tensorflow opencv-python numpy pillow
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

---

## 🖥️ Usage

### Basic Usage

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Select an Image**
   - Click the "SELECT IMAGE" button
   - Navigate to your image file
   - Supported formats: JPG, PNG, BMP

3. **View Results**
   - Green bounding boxes highlight detected digits
   - Predictions appear above each digit
   - Confidence scores shown in the results panel

### Best Practices for Accuracy

✅ **DO:**
- Use black ink on white paper
- Ensure good lighting (no harsh shadows)
- Write digits clearly with some spacing
- Keep digits reasonably sized (not too small)

❌ **AVOID:**
- Low-contrast images (gray pencil on gray paper)
- Touching or overlapping digits
- Extreme angles or distortion
- Very small digits (< 20px after processing)

### Example Images

```
Good Input:              Bad Input:
┌─────────────┐         ┌─────────────┐
│  1  2  3    │         │ 123456789   │  (touching)
│             │         │             │
│  4  5  6    │         │   1 2 3     │  (too light)
└─────────────┘         └─────────────┘
```

---

## 📂 Project Structure

```
digit-ocr-system/
│
├── main.py                  # Main application entry point
├── model.py                 # CNN model architecture
├── preprocess.py            # Image preprocessing pipeline
├── detector.py              # Digit detection logic
├── gui.py                   # Tkinter GUI interface
│
├── models/
│   └── digit_recognizer.h5  # Trained CNN weights
│
├── utils/
│   ├── image_utils.py       # Image manipulation helpers
│   └── visualization.py     # Bounding box drawing
│
├── tests/
│   ├── test_model.py        # Unit tests for model
│   └── test_pipeline.py     # Pipeline integration tests
│
├── sample_images/           # Example test images
│   ├── single_digit.jpg
│   ├── multiple_digits.jpg
│   └── noisy_image.jpg
│
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

---

## 🧪 Model Performance

### Training Metrics

- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Training Accuracy**: 99.2%
- **Validation Accuracy**: 98.7%
- **Test Accuracy**: 98.5%

### Real-World Performance

| Scenario | Accuracy | Notes |
|----------|----------|-------|
| Clean handwriting | 95-98% | Optimal conditions |
| Normal photos | 85-92% | Good lighting, clear writing |
| Challenging conditions | 70-80% | Poor lighting, noise, smudges |

### Confusion Matrix Insights

Most common misclassifications:
- **5 ↔ 3**: Similar curved shapes
- **7 ↔ 1**: Thin vertical strokes
- **8 ↔ 0**: Closed loops

---

## 🔬 Advanced Features

### Confidence Thresholding

The system only displays predictions with confidence > 70% by default. Adjust in `detector.py`:

```python
CONFIDENCE_THRESHOLD = 0.7  # Adjust between 0.5 and 0.95
```

### Custom Training

Retrain the model on your own dataset:

```bash
python train.py --dataset /path/to/data --epochs 20 --batch-size 32
```

### Batch Processing

Process multiple images at once:

```bash
python batch_process.py --input-dir ./images --output-dir ./results
```

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! Any contributions you make are **greatly appreciated**.

### How to Contribute

1. Fork the Project
2. Create your Feature Branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the Branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 .

# Format code
black .
```

---

## 🔮 Future Roadmap

### Short-term Goals
- [ ] Add support for alphabetical characters (A-Z)
- [ ] Implement confidence calibration
- [ ] Create web-based interface
- [ ] Add export to CSV/JSON functionality

### Long-term Vision
- [ ] Real-time webcam integration (mirror-corrected)
- [ ] Mathematical expression solver integration
- [ ] Multi-language support (Arabic, Chinese numerals)
- [ ] Mobile app development (Android/iOS)
- [ ] Cloud-based API deployment

---

## 🐛 Troubleshooting

### Common Issues

**Problem**: No digits detected
- **Solution**: Ensure sufficient contrast between digits and background
- **Solution**: Try adjusting the `CONTOUR_MIN_AREA` parameter

**Problem**: Too many false detections
- **Solution**: Increase `CONFIDENCE_THRESHOLD`
- **Solution**: Use cleaner paper with fewer smudges

**Problem**: Poor accuracy on certain digits
- **Solution**: Retrain model with more examples of those digits
- **Solution**: Ensure digits are not too stylized or cursive

**Problem**: Slow processing on large images
- **Solution**: Images are auto-downsampled, but you can manually reduce resolution
- **Solution**: Process smaller regions of interest

---

## 📚 References & Resources

### Papers
- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - LeCun et al.
- [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) - Krizhevsky et al.

### Datasets
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [EMNIST (Extended MNIST)](https://www.nist.gov/itl/products-and-services/emnist-dataset)

### Libraries
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **MNIST Dataset** creators for providing the foundational training data
- **TensorFlow** team for the deep learning framework
- **OpenCV** community for computer vision tools
- All contributors and users who help improve this project

---

## 👨‍💻 Author

**Your Name**
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/digit-ocr-system?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/digit-ocr-system?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/digit-ocr-system?style=social)

---

<div align="center">

**Made with 🧠 and 💻**

If you found this project helpful, consider giving it a ⭐!

[⬆ Back to Top](#-intelligent-digit-recognition-system-ocr)

</div>
