import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

class WordEmbedder:
    """
    Handles the training of Word2Vec word embeddings and generates 
    visualizations of the vector space using dimensionality reduction.
    """
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        """
        Initializes the Word2Vec model parameters.
        - vector_size: Dimensionality of the word vectors.
        - window: Maximum distance between the current and predicted word.
        - min_count: Ignores all words with total frequency lower than this.
        """
        self.model = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def train_word2vec(self, tokenized_documents):
        """
        Trains the Word2Vec model using the preprocessed documents.
        """
        print(f"Training Word2Vec model (vector_size={self.vector_size}, min_count={self.min_count})...")
        self.model = Word2Vec(
            sentences=tokenized_documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        vocab_size = len(self.model.wv.index_to_key)
        print(f"Model trained successfully. Vocabulary size: {vocab_size} words.")

    def plot_word_embeddings(self, output_path, top_n=50):
        """
        Plots the spatial distribution of the top N words using PCA 
        to reduce the vectors to 2 dimensions.
        """
        if self.model is None:
            print("Error: Model is not trained yet!")
            return

        # Extract the words and their vectors
        words = self.model.wv.index_to_key[:top_n]
        vectors = [self.model.wv[word] for word in words]

        # Reduce dimensions to 2D using Principal Component Analysis (PCA)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        # Create the scatter plot
        plt.figure(figsize=(14, 10))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, edgecolors='k', c='skyblue')

        # Annotate each point with its corresponding word
        for word, (x, y) in zip(words, vectors_2d):
            plt.text(x + 0.02, y + 0.02, word, fontsize=10, alpha=0.8)

        plt.title("Word2Vec Embeddings Distribution (PCA 2D Projection)", fontsize=16)
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save and show the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Embeddings plot saved successfully to: {output_path}")
        plt.show()
