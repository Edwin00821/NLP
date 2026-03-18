# Natural Language Technologies - NLP Pipeline

This repository contains the practical assignments for the Natural Language Technologies course, part of the Artificial Intelligence Engineering curriculum. The core objective of this project is to build a foundational Natural Language Processing (NLP) pipeline entirely from scratch, culminating in a Sentiment Analysis classifier.

## 📌 Project Overview

The project is divided into four main stages, applying fundamental NLP techniques to an opinions dataset. Notably, the vectorization steps (Term-Document Matrix and TF-IDF) are implemented manually using `pandas` and `numpy` to deeply understand the underlying math and data structures, without relying on automated feature extraction libraries like `sklearn.feature_extraction`.

### 🧪 Practices

* **Practice 1: Dataset Analysis & Cleaning**
  * Implementation of a text preprocessing pipeline: Tokenization, Stopwords removal, Punctuation removal, Lemmatization, and Stemming.
  * Exploratory Data Analysis (EDA) with statistical findings and visualizations.
  * Word cloud generation to visualize term frequencies.
* **Practice 2: Binary Term-Document Matrix**
  * Creation of a binary Term-Document Matrix (1s and 0s) representing word presence in the cleaned dataset.
  * *Constraint:* Built entirely from scratch using arrays/DataFrames. Stored as a CSV.
* **Practice 3: TF-IDF Implementation**
  * Custom algorithmic implementation of Term Frequency-Inverse Document Frequency (TF-IDF).
  * Transformation of the binary matrix into a weighted TF-IDF matrix.
* **Practice 4: Sentiment Classifier (Naive Bayes)**
  * Dataset labeling (Positive, Negative, Neutral).
  * Training a Naive Bayes classifier using the custom TF-IDF matrix.
  * Extraction and analysis of standard evaluation metrics.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                # Original, unmodified datasets
│   ├── processed/          # Cleaned datasets and generated matrices (CSVs)
│   └── labeled/            # Datasets with sentiment labels applied
├── src/                    # Source code for modular functions
│   ├── preprocessing.py    # Tokenization, stemming, lemmatization, etc.
│   ├── vectorization.py    # Custom TDM and TF-IDF functions
│   ├── visualization.py    # Word clouds and statistical plots
│   └── main.py             # Main execution script to run the practices
├── reports/                # PDF reports and evidence screenshots
├── ARCHITECTURE.md         # Detailed technical architecture of the pipeline
├── requirements.txt        # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed locally. The primary libraries used are `pandas`, `numpy`, `nltk`/`spacy` (for linguistic tasks), and `matplotlib`/`wordcloud` (for visualization).

### Installation & Execution

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Edwin00821/NLP.git
   ```

2. Navigate to the project directory and install the required dependencies:

   ```bash
   cd tln-nlp-pipeline
   pip install -r requirements.txt
   ```

3. Run the main pipeline script to execute the practices:

   ```bash
   python src/main.py
   ```
