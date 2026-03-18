import pandas as pd
import numpy as np
import os


class CustomVectorizer:
    """
    A custom implementation of text vectorization techniques (Term-Document Matrix and TF-IDF),
    built entirely with numpy and pandas without relying on sklearn feature_extraction.
    """

    def __init__(self):
        self.vocabulary = []
        self.vocab_index = {}

    def build_vocabulary(self, processed_documents):
        """Extracts the unique vocabulary from a list of tokenized documents."""
        unique_terms = set(
            token for doc in processed_documents for token in doc)
        self.vocabulary = sorted(list(unique_terms))
        self.vocab_index = {word: i for i, word in enumerate(self.vocabulary)}
        print(f"Vocabulary built with {len(self.vocabulary)} unique terms.")

    def fit_transform_binary(self, processed_documents, document_ids):
        """Creates a binary Term-Document Matrix (1 for presence, 0 for absence)."""
        if not self.vocabulary:
            self.build_vocabulary(processed_documents)

        num_docs = len(processed_documents)
        vocab_size = len(self.vocabulary)
        matrix = np.zeros((num_docs, vocab_size), dtype=int)

        for doc_idx, doc_tokens in enumerate(processed_documents):
            for token in doc_tokens:
                if token in self.vocab_index:
                    col_idx = self.vocab_index[token]
                    matrix[doc_idx, col_idx] = 1

        return pd.DataFrame(matrix, columns=self.vocabulary, index=document_ids)

    def fit_transform_tfidf(self, processed_documents, document_ids):
        """
        Creates a TF-IDF Term-Document Matrix.
        Uses raw count for TF and base-10 logarithm for IDF.
        """
        if not self.vocabulary:
            self.build_vocabulary(processed_documents)

        num_docs = len(processed_documents)
        vocab_size = len(self.vocabulary)

        # 1. Calculate Term Frequency (TF)
        # Initialize float matrix to store frequencies
        tf_matrix = np.zeros((num_docs, vocab_size), dtype=float)

        for doc_idx, doc_tokens in enumerate(processed_documents):
            for token in doc_tokens:
                if token in self.vocab_index:
                    col_idx = self.vocab_index[token]
                    tf_matrix[doc_idx, col_idx] += 1.0  # Increment raw count

        # 2. Calculate Inverse Document Frequency (IDF)
        # Calculate Document Frequency (DF): How many docs contain each term?
        # We find this by checking where tf_matrix is greater than 0, then summing down the columns
        df_array = np.sum(tf_matrix > 0, axis=0)

        # Calculate IDF: log10(N / DF). Added 1e-9 to prevent division by zero mathematically.
        idf_array = np.log10(num_docs / (df_array + 1e-9))

        # 3. Calculate Final TF-IDF
        # Multiply the TF matrix by the IDF array using numpy broadcasting
        tfidf_matrix = tf_matrix * idf_array

        return pd.DataFrame(tfidf_matrix, columns=self.vocabulary, index=document_ids)

    def save_matrix_to_csv(self, df, output_path):
        """Saves the DataFrame matrix to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Matrix saved successfully to: {output_path}")
