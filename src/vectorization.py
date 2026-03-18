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
        """
        Extracts the unique vocabulary from a list of tokenized documents.

        Args:
            processed_documents (list of lists): A list where each element is a list of clean tokens for a document.
        """
        # Flatten the list of lists and get unique terms using a set
        unique_terms = set(
            token for doc in processed_documents for token in doc)

        # Sort the vocabulary alphabetically for better readability
        self.vocabulary = sorted(list(unique_terms))

        # Create a dictionary mapping each word to its column index for fast lookups
        self.vocab_index = {word: i for i, word in enumerate(self.vocabulary)}
        print(f"Vocabulary built with {len(self.vocabulary)} unique terms.")

    def fit_transform_binary(self, processed_documents, document_ids):
        """
        Creates a binary Term-Document Matrix (1 for presence, 0 for absence).

        Args:
            processed_documents (list of lists): The tokenized documents.
            document_ids (list): The IDs or labels for each document to use as the index.

        Returns:
            pd.DataFrame: The binary Term-Document Matrix.
        """
        if not self.vocabulary:
            self.build_vocabulary(processed_documents)

        num_docs = len(processed_documents)
        vocab_size = len(self.vocabulary)

        # Initialize an empty matrix of zeros using numpy for efficiency
        matrix = np.zeros((num_docs, vocab_size), dtype=int)

        # Populate the matrix
        for doc_idx, doc_tokens in enumerate(processed_documents):
            for token in doc_tokens:
                if token in self.vocab_index:
                    col_idx = self.vocab_index[token]
                    matrix[doc_idx, col_idx] = 1  # Binary presence: set to 1

        # Convert the numpy array to a pandas DataFrame
        df_matrix = pd.DataFrame(
            matrix, columns=self.vocabulary, index=document_ids)
        return df_matrix

    def save_matrix_to_csv(self, df, output_path):
        """
        Saves the DataFrame matrix to a CSV file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Matrix saved successfully to: {output_path}")
