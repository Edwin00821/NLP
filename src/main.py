import os
import pandas as pd
from preprocessing import TextPreprocessor
from visualization import TextVisualizer
from vectorization import CustomVectorizer
from labeling import auto_label_dataset
from classifier import SentimentClassifier
from embeddings import WordEmbedder


def load_and_clean_data(data_path, preprocessor):
    df = pd.read_csv(data_path)
    text_column = df.columns[1]
    id_column = df.columns[0]
    df = df.dropna(subset=[text_column])

    processed_docs = [preprocessor.process_pipeline(
        str(text), use_lemmatization=True) for text in df[text_column]]
    doc_ids = df[id_column].tolist()

    return processed_docs, doc_ids


def run_practice_1(processed_docs, reports_dir):
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 1: DATA ANALYSIS & CLEANING")
    print("="*50)
    visualizer = TextVisualizer(output_dir=reports_dir)
    global_tokens = [token for doc in processed_docs for token in doc]

    if global_tokens:
        visualizer.plot_top_words(global_tokens, top_n=15)
        visualizer.generate_wordcloud(global_tokens)
        print("✅ Practice 1 visualizations completed!")


def run_practice_2(processed_docs, doc_ids, processed_dir):
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 2: BINARY TERM-DOCUMENT MATRIX")
    print("="*50)
    vectorizer = CustomVectorizer()
    binary_matrix_df = vectorizer.fit_transform_binary(processed_docs, doc_ids)

    print("\nExtract of the Matrix (First 10 rows, 5 random columns):")
    sample_cols = binary_matrix_df.columns[:5]
    print(binary_matrix_df[sample_cols].head(10))

    output_path = os.path.join(
        processed_dir, "binary_term_document_matrix.csv")
    vectorizer.save_matrix_to_csv(binary_matrix_df, output_path)
    print("✅ Practice 2 completed!")


def run_practice_3(processed_docs, doc_ids, processed_dir):
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 3: TF-IDF MATRIX")
    print("="*50)
    vectorizer = CustomVectorizer()

    print("[1/2] Calculating TF and IDF matrices...")
    tfidf_matrix_df = vectorizer.fit_transform_tfidf(processed_docs, doc_ids)

    print("\nExtract of the TF-IDF Matrix (First 10 rows, 5 mid-vocabulary columns):")
    # Picking columns from the middle of the vocabulary to show non-zero TF-IDF values
    mid_idx = len(tfidf_matrix_df.columns) // 2
    sample_cols = tfidf_matrix_df.columns[mid_idx:mid_idx+5]
    print(tfidf_matrix_df[sample_cols].head(10))

    print("\n[2/2] Saving TF-IDF matrix to CSV...")
    output_path = os.path.join(processed_dir, "tfidf_term_document_matrix.csv")
    vectorizer.save_matrix_to_csv(tfidf_matrix_df, output_path)
    print("✅ Practice 3 completed!")


def run_practice_4(data_path, labeled_dir, processed_docs, doc_ids):
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 4: SENTIMENT CLASSIFIER")
    print("="*50)

    # 1. Label the dataset
    labeled_path = os.path.join(labeled_dir, "dataset_Opiniones_labeled.csv")
    labeled_df = auto_label_dataset(data_path, labeled_path)

    # We need the labels corresponding to the document IDs we processed
    # Assuming IDs match the index order
    labels = labeled_df['sentimiento'].tolist()

    # 2. Get the TF-IDF matrix (Reusing Practice 3 logic silently to feed the model)
    print("\nExtracting TF-IDF features for training...")
    vectorizer = CustomVectorizer()
    tfidf_matrix_df = vectorizer.fit_transform_tfidf(processed_docs, doc_ids)

    # 3. Train and Evaluate Model
    classifier = SentimentClassifier()
    classifier.train_and_evaluate(tfidf_matrix_df, labels)
    print("\n✅ Practice 4 completed!")


def run_practice_5(processed_docs, reports_dir):
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 5: WORD2VEC EMBEDDINGS")
    print("="*50)

    # Initialize and train the embedder
    # Using min_count=2 so words that appear at least twice are included in the graph
    embedder = WordEmbedder(vector_size=100, window=5, min_count=2)
    embedder.train_word2vec(processed_docs)

    # Generate and save the distribution plot
    print("\nGenerating vector space distribution plot...")
    plot_path = os.path.join(reports_dir, "word2vec_distribution.png")

    # Plotting top 60 words for a clear visualization without too much overlapping
    embedder.plot_word_embeddings(plot_path, top_n=60)
    print("\n✅ Practice 5 completed!")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(BASE_DIR, "..")

    DATA_PATH = os.path.join(PROJECT_ROOT, "data",
                             "raw", "dataset_Opiniones.csv")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
    else:
        preprocessor = TextPreprocessor()
        print(f"Loading and processing dataset from: {DATA_PATH}")
        processed_docs, doc_ids = load_and_clean_data(DATA_PATH, preprocessor)

        # Uncomment the practices you want to run!
        # run_practice_1(processed_docs, REPORTS_DIR)
        # run_practice_2(processed_docs, doc_ids, PROCESSED_DIR)
        # run_practice_3(processed_docs, doc_ids, PROCESSED_DIR)
        # run_practice_4(DATA_PATH, LABELED_DIR, processed_docs, doc_ids)
        run_practice_5(processed_docs, REPORTS_DIR)
