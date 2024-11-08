import argparse
import logging
import yaml
import pandas as pd
from main.mongodb.MongoHelper import MongoDBClient
from main.mongodb.AnalysisMongoHelper import AnalysisMongoHelper
from main.analysis.TextAnalysisProcessor import TextAnalysisProcessor


class DataAnalyser:
    def __init__(self, config):
        """
        Initializes the DataAnalyser with dependencies from the configuration.

        Args:
            config (dict): Configuration dictionary loaded from YAML.
        """
        self.config = config
        self.setup_logging(config)

        # Initialize MongoDB Client and Analysis Helper
        mongo_uri = self._get_mongo_uri(config["mongo"]["uri_path"])
        self.mongo_client = MongoDBClient(
            uri=mongo_uri,
            database_name=config["mongo"]["database_name"],
            collection_name=config["mongo"]["collection_qualified"],
            test_mode=config["mongo"]["test_mode"]
        )
        self.analysis_helper = AnalysisMongoHelper(
            self.mongo_client, config["mongo"]["analysis_database_name"], config["mongo"]["analysis_collection_bigrams"]
        )

        # Load keyword sets for analysis
        tools, libraries, languages, skills, education = self.load_keywords(config["analysis"]["keywords_path"])

        # Initialize Text Analysis Processor
        self.text_processor = TextAnalysisProcessor(tools, libraries, languages, skills, education, self.analysis_helper)
        logging.info("DataAnalyser initialized successfully.")

    def setup_logging(self, config):
        """Sets up logging based on the provided configuration."""
        log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
        handlers = [logging.StreamHandler()] if config["logging"]["log_to_console"] else []
        handlers.append(logging.FileHandler(config["logging"]["log_file_analyse"]))

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        logging.info("Logging setup complete.")

    def _get_mongo_uri(self, uri_path: str) -> str:
        """Reads the MongoDB URI from a file and returns it."""
        try:
            with open(uri_path, "r") as file:
                return file.read().strip()
        except Exception as e:
            logging.error(f"Failed to read MongoDB URI from {uri_path}: {e}")
            raise

    def load_keywords(self, keywords_path: str):
        """
        Loads keywords from a CSV file and returns sets of tools, libraries, languages, skills, and education.

        Args:
            keywords_path (str): Path to the CSV file containing keywords.

        Returns:
            tuple: Sets for tools, libraries, languages, skills, and education.
        """
        logging.info(f"Loading keywords from {keywords_path}...")
        try:
            keywords_df = pd.read_csv(keywords_path, skiprows=1, header=None)
            keywords_df.columns = ["Tools", "Libraries", "Languages", "Skills", "Education"]

            tools = set(keywords_df['Tools'].dropna().str.strip().str.lower())
            libraries = set(keywords_df['Libraries'].dropna().str.strip().str.lower())
            languages = set(keywords_df['Languages'].dropna().str.strip().str.lower())
            skills = set(keywords_df['Skills'].dropna().str.strip().str.lower())
            education = set(keywords_df['Education'].dropna().str.strip().str.lower())

            logging.info("Keywords loaded successfully.")
            return tools, libraries, languages, skills, education
        except Exception as e:
            logging.error(f"Error loading keywords from {keywords_path}: {e}")
            raise

    def run_analysis(self):
        """Runs the data analysis and stores the results in MongoDB in batch."""
        logging.info("Starting data analysis...")

        try:
            default_role = self.config["analysis"].get("role")
            default_country = self.config["analysis"].get("country")
            default_state = self.config["analysis"].get("state")

            # Build the query dynamically
            query = {"country": default_country, "role": default_role}
            if default_state != "All":  # Only include 'state' in the query if it's not "All"
                query["state"] = default_state

            # Fetch documents based on the constructed query
            documents = self.text_processor.fetch_documents(query)

            if not documents:
                logging.warning("No documents to analyze.")
                return

            self.analysis_helper.initialize_index()

            analysis_results = []
            analysed_count = 0

            for doc in documents:
                data_df = pd.DataFrame(doc["qualified_text"], columns=["text"])

                result = self.text_processor.process_analysis(data_df, self.config["analysis"].get("bigram_thresh"))
                if result:
                    analysis_results.append(result)

                analysed_count += 1
                if analysed_count % 50 == 0:  # Log every 50 documents
                    logging.info(f"Analysed {analysed_count} documents...")

            # Store all results in batch
            self.analysis_helper.store_analysis_results_batch(default_role, default_country, default_state,
                                                              analysis_results)
            logging.info("Data analysis and storage complete.")
        except Exception as e:
            logging.error(f"Error during data analysis: {e}")
            raise


def main(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    data_analyser = DataAnalyser(config)
    data_analyser.run_analysis()

    data_analyser.mongo_client.close_connection()
    logging.info("MongoDB connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to analyze text data and store results in MongoDB.")
    parser.add_argument("config", help="YAML config file for execution", type=str)
    args = parser.parse_args()

    main(config_file=args.config)
