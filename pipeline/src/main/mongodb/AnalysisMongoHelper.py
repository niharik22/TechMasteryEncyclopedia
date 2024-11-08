import logging
from typing import Dict, List
from main.mongodb.MongoHelper import MongoDBClient


class AnalysisMongoHelper:
    def __init__(self, mongo_client: MongoDBClient, analysis_db: str, analysis_collection: str):
        """
        Helper class to interact with MongoDB for storing or updating analysis results.

        Args:
            mongo_client (MongoDBClient): Instance of the MongoDBClient class.
            analysis_db (str): Target database name for storing analysis.
            analysis_collection (str): Target collection name for storing analysis.
        """
        self.mongo_client = mongo_client
        self.analysis_db = analysis_db
        self.analysis_collection = analysis_collection

    def initialize_index(self):
        """Creates a unique index on role, country, and state if it doesn't already exist."""
        try:
            self.mongo_client.change_database_and_collection(self.analysis_db, self.analysis_collection)
            self.mongo_client.make_index([("role", 1), ("country", 1), ("state", 1)])
        except Exception as e:
            logging.error(f"Error creating index: {e}")

    def fetch_documents(self, query: dict = {}) -> List[Dict]:
        """
        Fetches documents from the source collection (e.g., linkedindb_prod -> qualified).

        Args:
            query (dict): Query to filter documents. Defaults to an empty query.

        Returns:
            List[Dict]: List of documents from the source collection.
        """
        try:
            results = self.mongo_client.query_documents(query)
            logging.info("Documents fetched successfully.")
            return list(results)
        except Exception as e:
            logging.error(f"Error fetching documents: {e}")
            return []

    def store_analysis_results_batch(self,  role: str, country: str, state: str, analysis_results: List[Dict]) -> bool:
        """
        Stores or updates bigram analysis results in batch.

        Args:
            role (str): The job role for the analysis.
            country (str): The country for the analysis.
            state (str): The state for the analysis.
            analysis_results (List[Dict]): List of analysis results to be inserted or updated.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        if not analysis_results:
            logging.warning("No analysis results to store.")
            return False

        try:
            # Change to the target collection once
            self.mongo_client.change_database_and_collection(self.analysis_db, self.analysis_collection)

            for result in analysis_results:
                # Prepare the query to locate the document
                query = {"role": role, "country": country, "state": state}

                # Fetch any existing document to update it if found
                existing_document = self.mongo_client.query_documents(query).limit(1)
                existing_document = next(existing_document, None)  # Get the first document or None

                if existing_document:
                    # Merge existing and new categories
                    merged_tools = self.merge_categories(existing_document.get("tools", []), result.get("tools", []))
                    merged_libraries = self.merge_categories(existing_document.get("libraries", []), result.get("libraries", []))
                    merged_languages = self.merge_categories(existing_document.get("languages", []), result.get("languages", []))
                    merged_skills = self.merge_categories(existing_document.get("skills", []), result.get("skills", []))
                    merged_education = self.merge_categories(existing_document.get("education", []), result.get("education", []))
                    merged_mixed = self.merge_categories(existing_document.get("mixed", []), result.get("mixed", []))

                    # Construct the updated document
                    updated_document = {
                        "role": role,
                        "country": country,
                        "state": state,
                        "tools": merged_tools,
                        "libraries": merged_libraries,
                        "languages": merged_languages,
                        "skills": merged_skills,
                        "education": merged_education,
                        "mixed": merged_mixed
                    }
                else:
                    # If no existing document, use the new result as-is
                    updated_document = result

                # Update the document with merged results using upsert=True
                self.mongo_client.update_document(query, updated_document, upsert=True)

            logging.info("All analysis results stored or updated successfully.")
            return True
        except Exception as e:
            logging.error(f"Error storing analysis results in batch: {e}")
            return False

    def merge_categories(self, existing: list, new: list) -> list:
        """Merges two lists of bigrams, avoiding duplicates, and summing scores for existing items."""
        merged = {tuple(item["bigram"]): item["score"] for item in existing}
        for item in new:
            bigram_tuple = tuple(item["bigram"])
            if bigram_tuple in merged:
                merged[bigram_tuple] += item["score"]
            else:
                merged[bigram_tuple] = item["score"]
        return [{"bigram": list(bigram), "score": score} for bigram, score in merged.items()]
