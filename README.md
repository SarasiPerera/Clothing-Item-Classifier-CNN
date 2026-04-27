# 👗 Clothing Item Classifier (Fashion-MNIST · LeNet-5 CNN)

A deep learning–based web application that classifies clothing items into 10 categories using a **LeNet-5 Convolutional Neural Network (CNN)** trained on the **Fashion-MNIST dataset**.
The system includes a complete ML pipeline and an interactive **Streamlit UI** for real-time predictions.

---

## 🚀 Project Overview

This project demonstrates the end-to-end workflow of a machine learning system:

* Data preprocessing and normalization
* CNN model design (LeNet-5 architecture)
* Model training and evaluation
* Deployment via an interactive web application

Users can upload an image of a clothing item, and the system predicts its category along with confidence scores.

---

## 🎯 Features

✔ Image upload and real-time classification
✔ Clean and modern Streamlit user interface
✔ Confidence score and probability distribution
✔ Preprocessing visualization (28×28 grayscale input)
✔ Modular and reusable code structure (`src/`)

---

## 🧠 Model Details

* **Architecture:** LeNet-5 CNN
* **Dataset:** Fashion-MNIST (70,000 grayscale images, 10 classes)
* **Input Size:** 28 × 28 grayscale
* **Output:** Softmax probabilities for 10 classes

### 📊 Classes

| Class | Description      |
| ----- | ---------------- |
| 0     | T-shirt / Top 👕 |
| 1     | Trouser 👖       |
| 2     | Pullover 🧥      |
| 3     | Dress 👗         |
| 4     | Coat 🧥          |
| 5     | Sandal 👡        |
| 6     | Shirt 👔         |
| 7     | Sneaker 👟       |
| 8     | Bag 👜           |
| 9     | Ankle Boot 🥾    |

---

## 📁 Project Structure

```
Clothing-Item-Classifier-CNN/

├── app/
│   └── app.py
├── notebook/
│   ├── fashion_mnist_lenet5.ipynb
├── data/
│   └── fashion_mnist/
├── models/
│   └── best_lenet5_fashion.keras
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Clothing-Item-Classifier-CNN.git
cd Clothing-Item-Classifier-CNN
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the model

```bash
python -m src.train
```

### 🔹 Evaluate the model

```bash
python -m src.evaluate
```

### 🔹 Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## 🖥️ Application Preview

*Add screenshots of your app here*

Example:

```
![App Screenshot](outputs/screenshot.png)
```

---

## 💡 How It Works

1. User uploads an image
2. Image is converted to grayscale and resized to 28×28
3. Pixel values are normalized
4. Processed image is fed into the CNN model
5. Model outputs class probabilities
6. Top prediction is displayed with confidence

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pillow (PIL)
* Streamlit

---

## 📌 Future Improvements

* Add drawing canvas input
* Improve model accuracy with deeper architectures
* Deploy application online (Streamlit Cloud / Hugging Face Spaces)
* Add real-time camera input

---

## 👩‍💻 Author

**Sarasi Perera**
Undergraduate – Faculty of Computing
University of Sri Jayewardenepura

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

* Fashion-MNIST dataset
* TensorFlow & Keras documentation
* Streamlit framework

---
