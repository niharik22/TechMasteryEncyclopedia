import logging
from typing import List, Dict
from main.mongodb.MongoHelper import MongoDBClient

class ClassifierMongoHelper:
    def __init__(self, mongo_client: MongoDBClient):
        """Initializes ClassifierMongoHelper with MongoDBClient instance"""
        self.mongo_client = mongo_client

    def fetch_data_for_classification(self) -> List[Dict]:
        """
        Fetches data from MongoDB for classification, retrieving only documents with 'cleaned_sentence' field.

        Returns:
            List[Dict]: List of documents with 'link' and 'cleaned_sentence' fields.
        """
        logging.info("Fetching data from MongoDB for classification...")

        try:
            # Fetch documents with 'cleaned_sentence' field and 'classified' equal to 0
            cursor = self.mongo_client.query_documents(
                query={
                    "$and": [
                        {"cleaned_sentence": {"$exists": True}},
                        {"classified": 0}
                    ]
                },
                projection={"link": 1, "cleaned_sentence": 1}
            )

            # Convert the cursor to a list for easier manipulation
            documents = list(cursor)

            if not documents:
                logging.warning("No documents found with 'cleaned_sentence' field.")
                return []

            logging.info(f"Fetched {len(documents)} documents for classification.")
            return documents

        except Exception as e:
            logging.error(f"Error fetching data for classification: {e}")
            return []

    def update_classified_documents(self, classified_documents: List[Dict]) -> None:
        """
        Batch updates MongoDB with classified sentences and sets 'classified' field to 1 for each document.

        Parameters:
            classified_documents (List[Dict]): List of documents with updated 'cleaned_sentence' and 'classified' fields.
        """
        logging.info(f"Updating classified sentences for {len(classified_documents)} documents.")
        for doc in classified_documents:
            link = doc["link"]
            update_data = {
                "cleaned_sentence": doc["cleaned_sentence"],
                "classified": 1  # Set classified to 1
            }
            update_query = {"link": link}

            try:
                # Perform the update in MongoDB
                self.mongo_client.update_document(query=update_query, update=update_data)
                logging.debug(f"Updated document for link: {link} with classified sentences and 'classified' set to 1.")
            except Exception as e:
                logging.error(f"Error updating document for link {link}: {e}")

        logging.info(f"Updated classified sentences.")