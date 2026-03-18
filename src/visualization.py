import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# Import the preprocessing class we created earlier
from preprocessing import TextPreprocessor


class TextVisualizer:
    """
    Handles Exploratory Data Analysis (EDA) and visualization for NLP text data.
    """

    def __init__(self, output_dir="../reports"):
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocessor = TextPreprocessor()

    def load_and_process_data(self, filepath):
        """
        Loads the dataset and applies the preprocessing pipeline to the text column.
        Returns a single list containing all clean tokens from all documents.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # Assuming the first column is 'Id' and the second is the text/opinion
        text_column = df.columns[1]

        # Drop rows with empty text
        df = df.dropna(subset=[text_column])

        all_tokens = []
        print("Processing text... (This might take a moment if lemmatization is active)")

        for text in df[text_column]:
            # Apply our NLP pipeline from preprocessing.py
            # Using lemmatization=False here if you don't have Spacy installed yet,
            # but True is recommended for better results.
            tokens = self.preprocessor.process_pipeline(
                str(text), use_lemmatization=False)
            all_tokens.extend(tokens)

        return all_tokens

    def plot_top_words(self, tokens, top_n=20):
        """
        Calculates word frequencies and generates a bar chart of the most common words.
        Saves the plot to the reports directory.
        """
        # Count frequencies
        word_counts = Counter(tokens)
        most_common = word_counts.most_common(top_n)

        # Separate words and counts for plotting
        words, counts = zip(*most_common)

        plt.figure(figsize=(12, 6))
        plt.bar(words, counts, color='skyblue', edgecolor='black')
        plt.title(f"Top {top_n} Most Frequent Words", fontsize=16)
        plt.xlabel("Words", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save and show
        save_path = os.path.join(self.output_dir, "term_frequencies.png")
        plt.savefig(save_path)
        print(f"Frequency plot saved successfully to: {save_path}")
        plt.show()

    def generate_wordcloud(self, tokens):
        """
        Generates and saves a WordCloud based on the processed tokens.
        """
        # WordCloud requires a single string
        text_for_cloud = " ".join(tokens)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text_for_cloud)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Opinions Word Cloud", fontsize=16)
        plt.tight_layout()

        # Save and show
        save_path = os.path.join(self.output_dir, "wordcloud.png")
        plt.savefig(save_path)
        print(f"WordCloud saved successfully to: {save_path}")
        plt.show()


# Example of execution
if __name__ == "__main__":
    # Adjust this path to where your CSV is located in your project
    # Example: "../data/raw/dataset_Opiniones.csv"
    dataset_path = "dataset_Opiniones.xlsx - Sheet1.csv"

    visualizer = TextVisualizer(output_dir="reports")

    try:
        # 1. Process data
        global_tokens = visualizer.load_and_process_data(dataset_path)

        if global_tokens:
            # 2. Generate Statistical Plot
            visualizer.plot_top_words(global_tokens, top_n=15)

            # 3. Generate Word Cloud
            visualizer.generate_wordcloud(global_tokens)
        else:
            print("No tokens were generated. Check your dataset and preprocessor.")

    except FileNotFoundError:
        print(
            f"Error: Could not find the file at {dataset_path}. Please check the path.")
