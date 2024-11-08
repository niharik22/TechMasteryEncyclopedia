import re
import string
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from main.mongodb.AnalysisMongoHelper import AnalysisMongoHelper


class TextAnalysisProcessor:
    def __init__(self, tools, libraries, languages, skills, education, mongo_helper: AnalysisMongoHelper):
        """
        Initializes the TextAnalysisProcessor, which handles text processing and MongoDB interaction.

        Args:
            tools (set): Set of tool keywords.
            libraries (set): Set of library keywords.
            languages (set): Set of language keywords.
            skills (set): Set of skill keywords.
            education (set): Set of education keywords.
            mongo_helper (AnalysisMongoHelper): Instance of AnalysisMongoHelper for database interaction.
        """
        self.stop = set(stopwords.words('english'))
        self.tools_pattern = self.create_pattern(tools)
        self.libraries_pattern = self.create_pattern(libraries)
        self.languages_pattern = self.create_pattern(languages)
        self.skills_pattern = self.create_pattern(skills)
        self.education_pattern = self.create_pattern(education)
        self.mongo_helper = mongo_helper

    def create_pattern(self, words_list):
        """Create a regex pattern from a list of words."""
        pattern = r'\b(?:' + '|'.join(map(re.escape, words_list)) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def cleanse_sentence(self, sentence: str) -> str:
        """Cleanse a sentence by removing punctuation, stopwords, and converting to lowercase."""
        sentence_clean = re.sub(r"[\n.,!?/\()-:]", " ", sentence)
        sentence_clean = re.sub(r"\s+", " ", sentence_clean).strip().lower()
        sentence_clean = " ".join(word for word in sentence_clean.split() if word not in self.stop)
        return sentence_clean

    def aggregate_bigrams(self, data_df: pd.DataFrame, thresh: int = 5):
        """Aggregate bigrams across the corpus and return their frequencies."""
        bigram_measures = BigramAssocMeasures()
        corpus_list = data_df['text'].drop_duplicates().dropna().apply(self.cleanse_sentence).tolist()
        corpus = ' '.join(corpus_list)
        finder = BigramCollocationFinder.from_words(corpus.split())
        finder.apply_freq_filter(thresh)
        return list(finder.ngram_fd.items())

    def get_category(self, bigram):
        """Determine the category of a bigram."""
        phrase = ' '.join(bigram)
        if self.skills_pattern.search(phrase):
            return "skills"
        elif self.libraries_pattern.search(phrase):
            return "libraries"
        elif self.languages_pattern.search(phrase):
            return "languages"
        elif self.tools_pattern.search(phrase):
            return "tools"
        elif self.education_pattern.search(phrase):
            return "education"

        word1, word2 = bigram
        if self.skills_pattern.search(word1) or self.skills_pattern.search(word2):
            return "skills"
        elif self.libraries_pattern.search(word1) or self.libraries_pattern.search(word2):
            return "libraries"
        elif self.languages_pattern.search(word1) or self.languages_pattern.search(word2):
            return "languages"
        elif self.tools_pattern.search(word1) or self.tools_pattern.search(word2):
            return "tools"
        elif self.education_pattern.search(word1) or self.education_pattern.search(word2):
            return "education"

        return None

    def categorize_bigrams(self, bigram_results):
        """Categorize bigrams into tools, libraries, languages, skills, education, or mixed."""
        categorized_bigrams = {
            "tools": [], "libraries": [], "languages": [], "skills": [], "education": [], "mixed": []
        }

        def clean_word(word):
            return word.lower().strip(string.punctuation)

        for bigram, score in bigram_results:
            bigram_cleaned = (clean_word(bigram[0]), clean_word(bigram[1]))
            category = self.get_category(bigram_cleaned)
            if category:
                categorized_bigrams[category].append({"bigram": bigram, "score": score})
            else:
                categorized_bigrams["mixed"].append({"bigram": bigram, "score": score})

        return categorized_bigrams

    def fetch_documents(self, query: dict = {}):
        """Fetch documents from the source collection."""
        try:
            results = self.mongo_helper.fetch_documents(query)
            logging.info("Documents fetched successfully.")
            return results
        except Exception as e:
            logging.error(f"Error fetching documents: {e}")
            return []

    def process_analysis(self, data_df: pd.DataFrame, thresh: int):
        """Perform bigram analysis and return the results."""
        try:
            bigram_results = self.aggregate_bigrams(data_df, thresh=thresh)
            categorized_bigrams = self.categorize_bigrams(bigram_results)
            return categorized_bigrams
        except Exception as e:
            logging.error(f"Error processing analysis: {e}")
            return {}
