from bs4 import BeautifulSoup
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy


class DescriptionCleaner:
    def __init__(self):
        """Initializes DescriptionCleaner with necessary NLP tools and custom stopwords."""
        try:
            self.lemmatizer = WordNetLemmatizer()
            custom_stopwords = {'ability', 'experience', 'communicate', 'demonstrate', 'empower', 'exercise', 'year',
                                'team', 'work'}
            self.stop_words = set(stopwords.words('english')).union(custom_stopwords)
            self.nlp = spacy.load("en_core_web_sm")  # Only for sentence splitting
            logging.info("DescriptionCleaner initialized with custom stopwords and NLP tools.")
        except Exception as e:
            logging.error(f"Error initializing DescriptionCleaner: {e}")
            raise

    def split_into_sentences(self, text: str) -> list:
        """
        Splits text into individual sentences, handling new lines and
        removing any leading hyphens.

        Args:
            text (str): Raw job description text.

        Returns:
            list: List of sentences.
        """
        try:
            logging.debug("Starting sentence splitting.")
            paragraphs = text.split('\n')
            sentences = []

            for paragraph in paragraphs:
                doc = self.nlp(paragraph)  # Process paragraph with spaCy
                for sent in doc.sents:
                    cleaned_sentence = sent.text.strip()
                    if cleaned_sentence.startswith('-'):
                        cleaned_sentence = cleaned_sentence[1:].strip()  # Remove leading hyphen
                    sentences.append(cleaned_sentence)

            logging.debug("Sentence splitting completed.")
            return sentences
        except Exception as e:
            logging.error(f"Error during sentence splitting: {e}")
            return []

    def clean_sentence(self, sentence: str) -> str:
        """
        Cleans a single sentence by removing HTML tags, unnecessary whitespace,
        numbers, special characters, stopwords, and applies lemmatization.

        Args:
            sentence (str): Individual sentence from job description.

        Returns:
            str: Cleaned sentence.
        """
        try:
            # 1. Remove HTML Tags
            soup = BeautifulSoup(sentence, "html.parser")
            cleaned_sentence = soup.get_text()

            # 2. Remove Unnecessary Line Breaks and Whitespace
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

            # 3. Lowercase Transformation
            cleaned_sentence = cleaned_sentence.lower()

            # 4. Remove Special Characters and Punctuation (retain colons, commas, and periods)
            cleaned_sentence = re.sub(r'[^\w\s.,:]', '', cleaned_sentence)

            # 5. Remove Numbers
            cleaned_sentence = re.sub(r'\d+', '', cleaned_sentence)

            # 6. Remove Stopwords
            cleaned_sentence = ' '.join(
                [word for word in cleaned_sentence.split() if word not in self.stop_words]
            )

            # 7. Lemmatization
            cleaned_sentence = ' '.join(
                [self.lemmatizer.lemmatize(word) for word in cleaned_sentence.split()]
            )

            return cleaned_sentence
        except Exception as e:
            logging.error(f"Error during sentence cleaning: {e}")
            return ""

    def process_description(self, description: str) -> list:
        """
        Combines sentence splitting and sentence cleaning, returning a list of cleaned sentences.
        """
        try:
            logging.info("Processing job description.")

            # Split description into sentences
            sentences = self.split_into_sentences(description)

            # Clean each sentence individually
            cleaned_sentences = [self.clean_sentence(sentence) for sentence in sentences]

            logging.info(f"Description processing completed with {len(cleaned_sentences)} sentences.")
            return cleaned_sentences
        except Exception as e:
            logging.error(f"Error during description processing: {e}")
            return []
