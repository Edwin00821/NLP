import os
import pandas as pd
from preprocessing import TextPreprocessor
from visualization import TextVisualizer
from vectorization import CustomVectorizer


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


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(BASE_DIR, "..")

    DATA_PATH = os.path.join(PROJECT_ROOT, "data",
                             "raw", "dataset_Opiniones.csv")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
    else:
        preprocessor = TextPreprocessor()
        print(f"Loading and processing dataset from: {DATA_PATH}")
        processed_docs, doc_ids = load_and_clean_data(DATA_PATH, preprocessor)

        # run_practice_1(processed_docs, REPORTS_DIR)
        # run_practice_2(processed_docs, doc_ids, PROCESSED_DIR)
        run_practice_3(processed_docs, doc_ids, PROCESSED_DIR)
