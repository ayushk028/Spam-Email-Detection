# Spam Email Detection using CNN

## 📌 Overview
This project implements a **Spam Email Detection** system using a **Convolutional Neural Network (CNN)**. The model classifies emails as either **Spam** or **Ham (Not Spam)** based on their content. Deep learning techniques are applied to enhance accuracy compared to traditional machine learning approaches.

## 📂 Dataset
The dataset used for training and evaluation consists of labeled emails categorized as spam or ham. It is preprocessed to extract text features and converted into numerical representations suitable for CNN processing.

### Data Preprocessing
- **Text Cleaning:** Removal of special characters, stopwords, and extra spaces.
- **Tokenization:** Conversion of text into sequences.
- **Padding:** Ensuring uniform input size for CNN.
- **Embedding Layer:** Word embeddings for improved feature extraction.

## 🔧 Technologies Used
- Python
- TensorFlow / Keras
- Natural Language Processing (NLP)
- CNN for Text Classification
- Jupyter Notebook / Google Colab

## 🏗 Model Architecture
The CNN model is structured as follows:
1. **Embedding Layer** - Converts words into dense vectors.
2. **Convolutional Layers** - Extracts spatial features from text sequences.
3. **Max-Pooling Layers** - Reduces dimensionality and retains important information.
4. **Fully Connected Layers** - Final decision-making layers using dense layers.
5. **Softmax Activation** - Outputs probability for spam/ham classification.

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/ayushk028/spam-email-detection-cnn.git
cd spam-email-detection-cnn
```
### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 3️⃣ Train the Model
```sh
python train.py
```
### 4️⃣ Test the Model
```sh
python test.py
```

## 📈 Performance & Evaluation
- **Accuracy:** Achieved an accuracy of ~90% on test data.
- **Loss Function:** Categorical Cross-Entropy.
- **Optimization Algorithm:** Adam Optimizer.
- **Evaluation Metrics:** Precision, Recall, F1-Score.

## 📊 Results
The model effectively differentiates between spam and non-spam emails, outperforming traditional ML models such as Naive Bayes and SVM in terms of accuracy and generalization.

## 🔥 Future Improvements
- Implement **Bidirectional LSTMs** to improve sequential understanding.
- Use **pre-trained word embeddings** (e.g., Word2Vec, GloVe) for better text representation.
- Deploy the model as a **web service or API**.

## 🤝 Contributing
Feel free to fork this repository and contribute! Pull requests are welcome.

## 📜 License
This project is open-source and available under the MIT License.

## 📬 Contact
For queries, reach out at [ayushk028.github.io](https://ayushk028.github.io).

