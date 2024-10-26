import logging
from main.mongodb.MongoHelper import MongoDBClient
from main.mongodb.PreprocessMongoHelper import MongoDataHelper
from main.preprocess.DescriptionCleaner import DescriptionCleaner
from typing import List
import argparse
import yaml

class DataPreprocessor:
    def __init__(self, mongo_client: MongoDBClient):
        """Initializes DataPreprocessor with an instance of MongoDBClient."""
        self.mongo_helper = MongoDataHelper(mongo_client)
        self.mongo_client = mongo_client
        self.cleaner = DescriptionCleaner()

    def retrieve_data(self):
        """
        Retrieves URLs and descriptions from MongoDB using MongoDataHelper.
        """
        logging.info("Retrieving data from MongoDB for preprocessing...")
        data = self.mongo_helper.get_urls_with_descriptions()
        logging.info(f"Retrieved {len(data)} records from MongoDB.")
        return data

    def process_data(self, data: dict) -> dict:
        """
        Processes data by cleaning each description and splitting into sentences.
        """
        logging.info("Starting data cleaning and sentence splitting process...")
        processed_data = {}

        for link, content in data.items():
            original_description = content.get("description", "")
            cleaned_sentences = self.cleaner.process_description(original_description)
            processed_data[link] = {
                "sentences": cleaned_sentences
            }

        logging.info(f"Data processing completed. {len(processed_data)} records processed.")
        return processed_data

    def save_processed_data(self, cleaned_data: dict, config) -> List[str]:
        """
        Saves the cleaned data back to MongoDB in a specified collection,
        storing all sentences for each link in a single document with an array of sentences.

        Returns:
            List[str]: List of links that were successfully saved to the MongoDB collection.
        """
        # Switch to the specified collection
        try:
            new_collection = config['mongo']['collection_clean']
            self.mongo_client.change_collection(new_collection_name=new_collection)
            logging.info(f"Switched to collection '{new_collection}'.")
        except Exception as e:
            logging.error(f"Error switching collection: {e}")
            return []

        logging.info("Saving processed data back to MongoDB...")

        # List to store links that were successfully saved
        good_links: List[str] = []

        for link, content in cleaned_data.items():
            sentences = content.get("sentences", [])

            # Prepare the document structure
            document = {
                "link": link,
                "cleaned_sentence": [{"text": sentence} for sentence in sentences]
            }

            try:
                # Use `update_document` with upsert to insert or update the document in MongoDB
                self.mongo_client.update_document(
                    {"link": link},  # Query to find existing document by link
                    document,  # Set the link and cleaned_sentence array
                    upsert=True  # If document doesn't exist, create it
                )
                # Append to good_links after a successful update
                good_links.append(link)
                logging.debug(f"Inserted/Updated document for link '{link}' with {len(sentences)} sentences.")
            except Exception as e:
                logging.error(f"Error inserting/updating document for link '{link}': {e}")

        logging.info(f"Cleaned data has been saved to MongoDB. Total successful links: {len(good_links)}")

        return good_links  # Return the list of successfully saved links

    def mark_links_as_processed(self, links: List[str], config) -> None:
        """
        Switches to the specified raw collection and updates the `processed` field to 1 for each link in the provided list.
        """
        # Switch to the specified raw collection
        try:
            raw_collection = config['mongo']['collection_raw']
            self.mongo_client.change_collection(new_collection_name=raw_collection)
            logging.info(f"Switched to collection '{raw_collection}'.")
        except Exception as e:
            logging.error(f"Error switching collection: {e}")
            return

        logging.info("Updating processed status for links...")

        for link in links:
            try:
                # Update the document with the specified link to set `processed` to 1
                self.mongo_client.update_document(
                    {"link": link},  # Query to find the document by link
                    {"processed": 1},  # Set processed field to 1
                    upsert=False  # Do not create a new document if the link doesn't exist
                )
                logging.debug(f"Marked link '{link}' as processed.")
            except Exception as e:
                logging.error(f"Error updating processed status for link '{link}': {e}")

        logging.info(f"Processed status updated for {len(links)} links.")

    def run_preprocess(self,config):
        """Main method to execute the entire preprocessing workflow."""
        data = self.retrieve_data()
        if not data:
            logging.warning("No data retrieved from MongoDB. Exiting preprocessing.")
            return

        cleaned_data = self.process_data(data)
        good_links = self.save_processed_data(cleaned_data,config)
        self.mark_links_as_processed(good_links,config)
        logging.info("Data preprocessing workflow completed.")


def main(config):
    # Setup logging based on the configuration
    setup_logging(config)

    # MongoDB client setup
    mongo_uri = open(config["mongo"]["uri_path"]).read().strip()
    try:
        mongo_client = MongoDBClient(
            uri=mongo_uri,
            database_name=config["mongo"]["database_name"],
            collection_name=config["mongo"]["collection_raw"],
            test_mode=config["mongo"]["test_mode"]
        )
    except Exception as e:
        logging.error(f"Error initializing MongoDB client: {e}")
        return  # Exit if MongoDB initialization fails

    # Initialize and run the DataPreprocessor
    preprocessor = DataPreprocessor(mongo_client)
    preprocessor.run_preprocess(config)

    # Close MongoDB connection after processing
    mongo_client.close_connection()


def setup_logging(config):
    """Sets up logging based on the provided configuration."""
    log_level = getattr(logging, config["logging"]["level"].upper(), logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["log_file_preprocess"]),
            logging.StreamHandler()
        ]
    )


if __name__ == "__main__":

    # Set up argparse to take the config file as a command-line argument
    parser = argparse.ArgumentParser(description="Script to preprocess data from MongoDB.")
    parser.add_argument("config", help="YAML config file for execution", type=str)
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config_file = yaml.safe_load(file)

    # Run the main function with the loaded configuration
    main(config=config_file)
