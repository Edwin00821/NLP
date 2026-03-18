import pandas as pd
import os


def auto_label_dataset(input_csv, output_csv):
    """
    Automatically labels the opinions dataset with 'positivo', 'negativo', or 'neutro'
    based on a heuristic keyword approach.
    """
    print(f"Reading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    text_col = df.columns[1]

    def assign_label(text):
        text = str(text).lower()
        # Keywords based on the Exploratory Data Analysis (Practice 1)
        positive_words = ['bonit', 'amor', 'chido', 'feliz', 'acierto',
                          'gusta', 'importante', 'oportunidad', 'amo', 'inclusión']
        negative_words = ['consumismo', 'comercial', 'gasto', 'odio', 'triste', 'presión',
                          'obligación', 'marketing', 'dinero', 'innecesario', 'hipocresía', 'capitalismo']

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            return 'positivo'
        elif neg_count > pos_count:
            return 'negativo'
        else:
            return 'neutro'

    print("Applying heuristic sentiment labeling...")
    df['sentimiento'] = df[text_col].apply(assign_label)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Labeled dataset saved successfully to: {output_csv}")

    return df
