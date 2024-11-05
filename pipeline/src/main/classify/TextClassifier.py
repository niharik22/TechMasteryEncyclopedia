import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np
import logging
import argparse
import yaml
from main.mongodb.MongoHelper import MongoDBClient
from main.mongodb.ClassifyMongoHelper import ClassifierMongoHelper


class TextClassifier:

    def __init__(self, model_path: str, tokenizer_path: str, mongo_client: MongoDBClient, max_length: int = 150):
        """Initializes the TextClassifier with the model, tokenizer, and MongoDB client."""
        logging.info("Initializing TextClassifier...")
        try:
            self.model = tf.saved_model.load(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            self.max_length = max_length
            self.mongo_helper = ClassifierMongoHelper(mongo_client)  # Instantiate the helper with MongoDB client
            logging.info("Model, tokenizer, and MongoDB client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing TextClassifier: {e}")
            raise

    def predict(self, text: str) -> int:
        """
        Tokenizes the input text, performs prediction, and returns the predicted class.

        Parameters:
        - text (str): Input text to classify.

        Returns:
        - int: Predicted class (0 or 1).
        """
        logging.debug("Tokenizing and predicting for text.")

        # Check for null or empty text
        if not text:
            logging.debug("Empty or null input text. Returning default class 0.")
            return 0

        try:
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="tf"
            )
            infer = self.model.signatures["serving_default"]
            output = infer(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            logits = output['logits'][0]
            probabilities = tf.nn.softmax(logits).numpy()
            predicted_class = int(np.argmax(probabilities))
            logging.debug(f"Predicted class: {predicted_class} for text.")
            return predicted_class
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return 0  # Defaulting to 0 in a failure in prediction

    def classify_and_update_mongo(self):
        """
        Fetches records from MongoDB, classifies each text in 'cleaned_sentence', and updates MongoDB.
        """
        logging.info("Starting classification and MongoDB update process...")

        # Fetch data from MongoDB
        documents = self.mongo_helper.fetch_data_for_classification()
        if not documents:
            logging.warning("No documents to classify.")
            return

        logging.info(f"Starting the Classification for documents.")

        # Initialize counter for logging progress in batches of 10
        classified_documents = []
        classified_count = 0

        # Classify each sentence in all documents and store the results
        for doc in documents:
            link = doc.get("link")
            cleaned_sentences = doc.get("cleaned_sentence", [])

            logging.debug(f"Classifying sentences for document with link: {link}")

            # Classify each sentence and add 'qualified' field
            classified_sentences = [{"text": sentence["text"], "qualified": self.predict(sentence["text"])}
                                    for sentence in cleaned_sentences]

            # Prepare document for batch update
            classified_documents.append({
                "link": link,
                "cleaned_sentence": classified_sentences
            })

            # Increment counter and log every 10 documents processed
            classified_count += 1
            if classified_count % 10 == 0:
                logging.info(f"{classified_count} documents classified and ready for update.")

        logging.info(f"Classification for documents completed.")

        # Batch update MongoDB with classified documents
        try:
            self.mongo_helper.update_classified_documents(classified_documents)
            logging.info(f"Completed classification and MongoDB update for {classified_count} documents.")
        except Exception as e:
            logging.error(f"Error updating MongoDB with classified documents: {e}")


def setup_logging(config):
    """Sets up logging based on the provided configuration."""
    log_level = getattr(logging, config["logging"]["level"].upper(), logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["log_file_classifier"]),
            logging.StreamHandler()
        ]
    )


def main(config):
    # Setup logging based on the configuration
    setup_logging(config)

    # Load model and tokenizer paths from config
    model_path = config["bert"]["classifier"]["model_path"]
    tokenizer_path = config["bert"]["classifier"]["tokenizer_path"]

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

    # Initialize TextClassifier and classify data
    classifier = TextClassifier(model_path, tokenizer_path, mongo_client)
    classifier.classify_and_update_mongo()

    # Close MongoDB connection after processing
    mongo_client.close_connection()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to classify text data from MongoDB.")
    parser.add_argument("config", help="YAML config file for execution", type=str)
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config_file = yaml.safe_load(file)

    # Run the main function with the loaded configuration
    main(config=config_file)
