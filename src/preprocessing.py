import string
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# Download necessary NLTK datasets (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


class TextPreprocessor:
    """
    A comprehensive text preprocessing pipeline for Natural Language Processing,
    optimized for Spanish text.
    """

    def __init__(self):
        # Load Spanish stemmer
        self.stemmer = SnowballStemmer("spanish")

        # Load Spacy model for Spanish lemmatization
        # (Requires: python -m spacy download es_core_news_sm)
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            print(
                "Warning: Spacy model 'es_core_news_sm' not found. Lemmatization might fail.")
            self.nlp = None

        # Base NLTK stopwords + custom provided stopwords
        base_stopwords = set(stopwords.words('spanish'))
        custom_stopwords = {
            "actualmente", "adelante", "además", "afirmó", "agregó", "ahora", "ahí", "al", "algo",
            "alguna", "algunas", "alguno", "algunos", "algún", "alrededor", "ambos", "ampleamos",
            "ante", "anterior", "antes", "apenas", "aproximadamente", "aquel", "aquellas", "aquellos",
            "aqui", "aquí", "arriba", "aseguró", "así", "atras", "aunque", "ayer", "añadió", "aún",
            "bajo", "bastante", "bien", "buen", "buena", "buenas", "bueno", "buenos", "cada", "casi",
            "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco", "comentó", "como", "con",
            "conocer", "conseguimos", "conseguir", "considera", "consideró", "consigo", "consigue",
            "consiguen", "consigues", "contra", "cosas", "creo", "cual", "cuales", "cualquier",
            "cuando", "cuanto", "cuatro", "cuenta", "cómo", "da", "dado", "dan", "dar", "de", "debe",
            "deben", "debido", "decir", "dejó", "del", "demás", "dentro", "desde", "después", "dice",
            "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
            "dos", "durante", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos", "embargo",
            "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra",
            "entonces", "entre", "era", "erais", "eramos", "eran", "eras", "eres", "es", "esa", "esas",
            "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban", "estabas", "estad",
            "estada", "estadas", "estado", "estados", "estais", "estamos", "estan", "estando",
            "estar", "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis", "estaría",
            "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto",
            "estos", "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras",
            "estuvieron", "estuviese", "estuvieseis", "estuviesen", "estuvieses", "estuvimos",
            "estuviste", "estuvisteis", "estuviéramos", "estuviésemos", "estuvo", "está", "estábamos",
            "estáis", "están", "estás", "esté", "estéis", "estén", "estés", "ex", "existe", "existen",
            "explicó", "expresó", "fin", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron",
            "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis",
            "fuéramos", "fuésemos", "gran", "grandes", "gueno", "ha", "haber", "habida", "habidas",
            "habido", "habidos", "habiendo", "habremos", "habrá", "habrán", "habrás", "habré",
            "habréis", "habría", "habríais", "habríamos", "habrían", "habrías", "habéis", "había",
            "habíais", "habíamos", "habían", "habías", "hace", "haceis", "hacemos", "hacen", "hacer",
            "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "has", "hasta", "hay", "haya",
            "hayamos", "hayan", "hayas", "hayáis", "he", "hecho", "hemos", "hicieron", "hizo", "hoy",
            "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese", "hubieseis",
            "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis", "hubiéramos", "hubiésemos",
            "hubo", "igual", "incluso", "indicó", "informó", "intenta", "intentais", "intentamos",
            "intentan", "intentar", "intentas", "intento", "ir", "junto", "la", "lado", "largo",
            "las", "le", "les", "llegó", "lleva", "llevar", "lo", "los", "luego", "lugar", "manera",
            "manifestó", "mayor", "me", "mediante", "mejor", "mencionó", "menos", "mi", "mientras",
            "mio", "mis", "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas",
            "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "nada", "nadie",
            "ni", "ninguna", "ningunas", "ninguno", "ningunos", "ningún", "no", "nos", "nosotras",
            "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva", "nuevas", "nuevo",
            "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro", "otros", "para", "parece",
            "parte", "partir", "pasada", "pasado", "pero", "pesar", "poca", "pocas", "poco", "pocos",
            "podeis", "podemos", "poder", "podria", "podriais", "podriamos", "podrian", "podrias",
            "podrá", "podrán", "podría", "podrían", "poner", "por", "por qué", "porque", "posible",
            "primer", "primera", "primero", "primeros", "principalmente", "propia", "propias",
            "propio", "propios", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
            "pues", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quién", "qué",
            "realizado", "realizar", "realizó", "respecto", "sabe", "sabeis", "sabemos", "saben",
            "saber", "sabes", "se", "sea", "seamos", "sean", "seas", "segunda", "segundo", "según",
            "seis", "ser", "seremos", "será", "serán", "serás", "seré", "seréis", "sería", "seríais",
            "seríamos", "serían", "serías", "seáis", "señaló", "si", "sido", "siempre", "siendo",
            "siete", "sigue", "siguiente", "sin", "sino", "sobre", "sois", "sola", "solamente",
            "solas", "solo", "solos", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo",
            "suyos", "sí", "sólo", "tal", "también", "tampoco", "tan", "tanto", "te", "tendremos",
            "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais", "tendríamos",
            "tendrían", "tendrías", "tened", "teneis", "tenemos", "tener", "tenga", "tengamos",
            "tengan", "tengas", "tengo", "tengáis", "tenida", "tenidas", "tenido", "tenidos",
            "teniendo", "tenéis", "tenía", "teníais", "teníamos", "tenían", "tenías", "tercera",
            "ti", "tiempo", "tiene", "tienen", "tienes", "toda", "todas", "todavía", "todo", "todos",
            "total", "trabaja", "trabajais", "trabamos", "trabajan", "trabajar", "trabajas",
            "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuve", "tuviera", "tuvierais",
            "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses",
            "tuvimos", "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo", "tuya", "tuyas",
            "tuyo", "tuyos", "tú", "ultimo", "un", "una", "unas", "uno", "unos", "usa", "usais",
            "usamos", "usan", "usar", "usas", "uso", "usted", "va", "vais", "valor", "vamos", "van",
            "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
            "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "y", "ya",
            "yo", "él", "éramos", "ésta", "éstas", "éste", "éstos", "última", "últimas", "último",
            "últimos"
        }
        self.stop_words = base_stopwords.union(custom_stopwords)

    def to_lowercase(self, text):
        """Converts text to lowercase."""
        return str(text).lower()

    def remove_accents(self, text):
        """Replaces accented characters with their unaccented equivalents."""
        replacements = (
            ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")
        )
        for a, b in replacements:
            text = text.replace(a, b)
        return text

    def remove_punctuation(self, text):
        """Removes punctuation and special characters from the text."""
        forbidden = {
            "?", "¿", "¡", "!", "'", '"', "‘", "’", "“", "”", "<", ">", "(", ")",
            ".", ",", ":", ";", "-", "&", "@", "/", "N/A", "#", "$", "´", "`",
            "«", "»", "—", "–", "…", "•"
        }
        all_punctuation = set(string.punctuation).union(forbidden)
        # Remove punctuation
        text_no_punct = "".join(
            [char for char in text if char not in all_punctuation])
        # Replace multiple spaces with a single space
        return " ".join(text_no_punct.split())

    def tokenize_text(self, text):
        """Splits text into a list of words (tokens)."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Removes stop words from a list of tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def apply_stemming(self, tokens):
        """Applies Snowball Stemmer to a list of tokens."""
        return [self.stemmer.stem(word) for word in tokens]

    def apply_lemmatization(self, text):
        """
        Applies Lemmatization using Spacy. 
        Note: Spacy expects a string, not a list of tokens.
        """
        if self.nlp is None:
            return text.split()  # Fallback if spacy is not installed

        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def process_pipeline(self, text, use_lemmatization=True):
        """
        Executes the full preprocessing pipeline on a single string of text.
        Returns a list of clean tokens.
        """
        text = self.to_lowercase(text)
        text = self.remove_accents(text)
        text = self.remove_punctuation(text)

        if use_lemmatization and self.nlp:
            # Lemmatization works better on the full string before custom tokenization
            tokens = self.apply_lemmatization(text)
            tokens = self.remove_stopwords(tokens)
        else:
            tokens = self.tokenize_text(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.apply_stemming(tokens)

        return tokens


# Example of usage:
if __name__ == "__main__":
    processor = TextPreprocessor()
    sample_text = "¡Qué opinas de la celebración del 14 de febrero en el sentido de ser nombrado 'El día del amor y la amistad'?"

    # Process using Lemmatization (Recommended for Sentiment Analysis)
    clean_tokens = processor.process_pipeline(
        sample_text, use_lemmatization=True)
    print("Lemmatized Tokens:", clean_tokens)
