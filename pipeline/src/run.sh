#!/bin/bash

# Run the scraping script with the config file

#python -m main.scrape.LinkedIn_Scraper utilities/config.docker.datascientist.usa.yaml

python -m main.preprocess.DataPreprocessor utilities/config.docker.datascientist.usa.yaml