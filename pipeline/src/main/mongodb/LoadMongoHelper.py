from main.mongodb.MongoHelper import MongoDBClient
import logging
from typing import List, Dict

class LoadMongoHelper:
    def __init__(self, mongo_client: MongoDBClient):
        self.mongo_client = mongo_client
        self.combined_entries = []  # Collects entries for bulk insertion

    def fetch_clean_data(self) -> List[Dict]:
        """Fetches documents in 'clean' collection with 'combined: 0' and 'classified: 1'."""
        logging.info("Fetching data from 'clean' collection.")

        # Fetch the documents
        docs = self.mongo_client.query_documents(
            query={"combined": 0, "classified": 1},
            projection={"link": 1, "cleaned_sentence": 1},
            col_name="clean"
        )

        # Log the number of fetched documents
        logging.info(f"Fetched {len(list(docs.clone()))} documents from 'clean' collection.")

        return docs

    def combine_and_collect(self, clean_doc: Dict) -> None:
        """Combines data from 'raw' and 'clean' collections and adds it to the combined_entries list."""
        logging.info("Starting combine and collect...")
        link = clean_doc["link"]

        # Fetch corresponding data from 'raw'
        raw_doc = self.mongo_client.query_documents(
            query={"link": link},
            projection={"place_of_work": 1, "city": 1, "country": 1, "state": 1, "role": 1, "scraped_date": 1},
            col_name="raw"
        )

        if raw_doc:
            # Filter qualified sentences
            qualified_sentences = [
                sentence["text"]
                for sentence in clean_doc["cleaned_sentence"]
                if sentence.get("qualified") == 1
            ]

            # Prepare the combined entry
            combined_entry = {
                "link": link,
                "city": raw_doc[0].get("city"),
                "country": raw_doc[0].get("country"),
                "state": raw_doc[0].get("state"),
                "place_of_work": raw_doc[0].get("place_of_work"),
                "role": raw_doc[0].get("role"),
                "scraped_date": raw_doc[0].get("scraped_date"),
                "qualified_text": qualified_sentences
            }

            # Add to combined entries list
            self.combined_entries.append(combined_entry)
            logging.debug(f"Combined data prepared for link: {link}")

    def bulk_insert_and_update(self) -> None:
        """Inserts all collected combined entries into 'qualified' collection in bulk and updates 'combined' flag."""
        logging.info("Starting bulk insert and update...")
        if self.combined_entries:
            logging.info(f"Loading {len(self.combined_entries)} documents.")
            # Perform bulk insert through MongoDBClient
            if self.mongo_client.insert_documents(self.combined_entries, col_name="qualified"):
                # Update 'combined' field for each processed link in 'clean'
                links_to_update = [entry["link"] for entry in self.combined_entries]
                self.mongo_client.update_many_documents(
                    query={"link": {"$in": links_to_update}},
                    update={"combined": 1},
                    col_name="clean"
                )

            # Clear the list after insertion and update
            self.combined_entries.clear()
