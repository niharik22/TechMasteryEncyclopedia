from pymongo import MongoClient
import logging


class MongoDBClient:
    def __init__(self, uri: str, database_name: str, collection_name: str, test_mode: bool):
        try:
            self.client = MongoClient(uri)
            if test_mode:
                self.db = self.client['test_db']  # Use test database if in test mode
            else:
                self.db = self.client[database_name]  # Use production database
            self.collection = self.db[collection_name]
            logging.info(f"Connected to MongoDB database: {self.db.name}, collection: {self.collection.name}")
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            raise

    def make_index(self, index: str) -> None:
        """Creates an index for the specified field in the collection."""
        try:
            self.collection.create_index(index, unique=True)
            logging.info(f"Index created on {index}")
        except Exception as e:
            logging.error(f"Error creating index: {e}")

    def insert_document(self, doc: dict, col_name: str = None) -> bool:
        """Inserts a document into the collection."""
        collection = self.collection if col_name is None else self.db[col_name]
        try:
            collection.insert_one(doc)
            logging.info(f"Document inserted into {collection.name}")
            return True
        except Exception as e:
            logging.error(f"Error inserting document: {e}")
            return False

    def update_document(self, query: dict, update: dict, upsert: bool = True, col_name: str = None):
        """Updates or inserts a document based on the query."""
        collection = self.collection if col_name is None else self.db[col_name]
        try:
            collection.update_one(query, {'$set': update}, upsert=upsert)
            logging.debug(f"Document updated or inserted in {collection.name}")
        except Exception as e:
            logging.error(f"Error updating document: {e}")

    def query_documents(self, query: dict, projection: dict = None, col_name: str = None):
        """Queries documents from the collection."""
        collection = self.collection if col_name is None else self.db[col_name]
        try:
            results = collection.find(query, projection)
            logging.info(f"Queried documents from {collection.name}")
            return results
        except Exception as e:
            logging.error(f"Error querying documents: {e}")
            return None

    def close_connection(self) -> None:
        """Closes the connection to MongoDB."""
        try:
            self.client.close()
            logging.info("MongoDB connection closed.")
        except Exception as e:
            logging.error(f"Error closing MongoDB connection: {e}")