# 🎭 Face and Emotion Detection using CNN

A real-time facial emotion recognition system built using **TensorFlow/Keras** and **OpenCV**. The project detects faces from images or a live webcam feed and classifies facial expressions into seven emotion categories using a Convolutional Neural Network (CNN).

---

## ✨ Features

- 😊 Real-time webcam emotion detection
- 🖼️ Emotion prediction from static images
- 👤 Face detection using OpenCV Haar Cascade
- 🧠 Custom CNN trained on the FER2013 dataset
- 📈 Modular training pipeline with callbacks
- 💾 Automatic model checkpointing
- ⚙️ Clean and modular Python project structure

---

## 📌 Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📂 Project Structure

```
face_and_emotion_detection/
│
├── models/
│   ├── emotion_model.keras
│   ├── haarcascade_frontalface_default.xml
│   └── model_v6_23.hdf5
│
├── notebooks/
│   └── EmotionDetector_v2.ipynb
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── emotion_detector.py
│   ├── face_detector.py
│   ├── main.py
│   ├── train.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── webcam.py
│
├── test_images/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
cd face_and_emotion_detection
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the **FER2013 Facial Expression Recognition Dataset**.

Download the dataset from Kaggle and extract it as:

```
fer2013/
    train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

    validation/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
```

---

## 🏋️ Training

Train the CNN using

```bash
python src/train.py
```

The best performing model is automatically saved to

```
models/emotion_model.keras
```

using `ModelCheckpoint`.

---

## 📷 Run Webcam Emotion Detection

```bash
python src/main.py --webcam
```

Press **Q** to exit.

---

## 🖼️ Predict Emotion from an Image

```bash
python src/main.py --image test_images/example.jpg
```

---

## 🧠 Model Architecture

The model consists of:

- Convolutional Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Dropout Regularization
- Fully Connected Dense Layer
- Softmax Output Layer

---

## 📈 Performance

- Dataset: FER2013
- Classes: 7
- Input Size: 48×48 (Grayscale)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Data Augmentation: Rotation, Zoom, Shear, Horizontal Flip

**Best Validation Accuracy**

```
57.68%
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 🔮 Future Improvements

- Improve prediction stability using temporal smoothing
- Replace Haar Cascade with MediaPipe Face Detection
- Deploy as a Flask web application
- Export the trained model to TensorFlow Lite
- Improve accuracy using Mini-Xception or EfficientNet

---

## 📜 License

This project is intended for educational and research purposes.