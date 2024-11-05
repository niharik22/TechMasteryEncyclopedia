import logging
from main.mongodb.MongoHelper import MongoDBClient


class MongoDataHelper:
    def __init__(self, mongo_client: MongoDBClient):
        """Initializes with an instance of MongoDBClient."""
        self.mongo_client = mongo_client

    def get_urls_with_descriptions(self) -> dict:
        """
        Retrieves URLs and their respective descriptions as a dictionary
        """

        # Query to filter documents where 'processed' is 0 and 'description' field exists
        query = {
            "processed": 0,
            "description": {"$exists": True}
        }

        # Projection to include only the `link` and `description` fields, excluding `_id`
        projection = {"link": 1, "description": 1, "_id": 0}

        try:
            # Using MongoDBClient's `query_documents` to fetch all documents
            results = self.mongo_client.query_documents(query=query, projection=projection)

            # Check if results are returned and build the dictionary
            if results:
                url_description_dict = {
                    doc["link"]: {
                        "description": doc["description"]
                    } for doc in results
                }
                if url_description_dict:
                    logging.info(f"Fetched {len(url_description_dict)} records from the collection.")
                    return url_description_dict
                else:
                    # Log a warning if query executed but returned no matching documents
                    logging.warning("Query executed, but no matching documents found in the collection.")
                    return {}
            else:
                # If `results` is None, log a warning about potential collection issues
                logging.warning("No documents returned; the collection might be empty or query is invalid.")
                return {}

        except Exception as e:
            # Log any exceptions raised during the query process with a detailed error message
            logging.error(f"An unexpected error occurred while querying the database: {e}")
            return {}