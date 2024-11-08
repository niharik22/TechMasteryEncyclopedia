#!/bin/bash

# Run the scraping script with the config file

python -m main.scrape.LinkedIn_Scraper utilities/config.docker.se.canada.yaml

python -m main.preprocess.DataPreprocessor utilities/config.docker.se.canada.yaml

python -m main.classify.TextClassifier utilities/config.docker.se.canada.yaml

python -m main.load.DataLoad utilities/config.docker.se.canada.yaml

python -m main.analysis.DataAnalyser utilities/config.docker.se.canada.yaml