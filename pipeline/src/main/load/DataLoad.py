import logging
import argparse
import yaml
from typing import List, Dict
from main.mongodb.MongoHelper import MongoDBClient
from main.mongodb.LoadMongoHelper import LoadMongoHelper

class Load:
    def __init__(self, helper: LoadMongoHelper):
        self.helper = helper

    def process_all_data(self):
        """Processes all documents in 'clean' with 'combined: 0'."""
        logging.info("Starting data processing...")
        clean_docs = self.helper.fetch_clean_data()

        if not clean_docs or len(list(clean_docs.clone())) == 0:
            logging.info("No unprocessed documents found, data loading complete")
            return

        for doc in clean_docs:
            self.helper.combine_and_collect(doc)

        # Perform a bulk insert and update in a single call
        self.helper.bulk_insert_and_update()

        logging.info("Data loading completed.")


def setup_logging(config):
    """Sets up logging based on the provided configuration."""
    log_level = getattr(logging, config["logging"]["level"].upper(), logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["log_file_load"]),
            logging.StreamHandler()
        ]
    )


def main(config):
    # Setup logging based on the configuration
    setup_logging(config)

    # MongoDB client setup
    mongo_uri = open(config["mongo"]["uri_path"]).read().strip()
    try:
        mongo_client = MongoDBClient(
            uri=mongo_uri,
            database_name=config["mongo"]["database_name"],
            collection_name=config["mongo"]["collection_clean"],
            test_mode=config["mongo"]["test_mode"]
        )
    except Exception as e:
        logging.error(f"Error initializing MongoDB client: {e}")
        return  # Exit if MongoDB initialization fails

    # Initialize LoadMongoHelper and Load, then process data
    load_helper = LoadMongoHelper(mongo_client)
    load = Load(load_helper)
    load.process_all_data()

    # Close MongoDB connection after processing
    mongo_client.close_connection()


if __name__ == "__main__":
    # Set up argument parser for YAML config file
    parser = argparse.ArgumentParser(description="Script to process and combine data from MongoDB.")
    parser.add_argument("config", help="YAML config file for execution", type=str)
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config_file = yaml.safe_load(file)

    # Run the main function with the loaded configuration
    main(config=config_file)
