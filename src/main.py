import os
from preprocessing import TextPreprocessor
from visualization import TextVisualizer


def run_practice_1(data_path, reports_dir):
    """
    Executes the pipeline for Practice 1: Analysis and Cleaning.
    """
    print("\n" + "="*50)
    print("🚀 STARTING PRACTICE 1: DATA ANALYSIS & CLEANING")
    print("="*50)

    # Initialize the visualizer (which includes the preprocessor)
    visualizer = TextVisualizer(output_dir=reports_dir)

    # 1. Load and process the data
    print(f"\n[1/3] Loading and cleaning dataset from: {data_path}")
    tokens = visualizer.load_and_process_data(data_path)

    if tokens:
        print(f"      -> Successfully extracted {len(tokens)} clean tokens.")

        # 2. Generate Statistical Plot
        print("\n[2/3] Generating statistical analysis (Top Words)...")
        visualizer.plot_top_words(tokens, top_n=15)

        # 3. Generate Word Cloud
        print("\n[3/3] Generating Word Cloud...")
        visualizer.generate_wordcloud(tokens)

        print("\n✅ Practice 1 completed successfully!")
        print(f"📊 Check the '{reports_dir}' folder for the output images.")
    else:
        print("\n❌ Pipeline failed to generate tokens. Please check your dataset.")


if __name__ == "__main__":
    # Define absolute paths based on the location of this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(BASE_DIR, "..")

    # Define input and output paths
    # Note: Rename your CSV to 'dataset_Opiniones.csv' and place it in 'data/raw/'
    DATA_PATH = os.path.join(PROJECT_ROOT, "data",
                             "raw", "dataset_Opiniones.csv")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

    # Check if the dataset exists before running
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please ensure your CSV file is inside the 'data/raw/' directory.")
    else:
        run_practice_1(DATA_PATH, REPORTS_DIR)
