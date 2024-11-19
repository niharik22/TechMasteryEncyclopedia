# Tech Mastery Encyclopedia - Classifier Modeling & Text Analysis

This document outlines the methodology used in the Classifier Modeling and Text Analysis steps of the Tech Mastery Encyclopedia project, which processes and classifies job descriptions to extract structured insights. This README provides a detailed breakdown of my approach and models used, including Naive Bayes, SVM, LSTM, and BERT classifiers, as well as bigram analysis using NLP techniques.

---

## Table of Contents

1. [Data Collection - Scraping](#data-collection---scraping)
   - [Overview](#overview)
   - [Scraping Workflow Walkthrough](#scraping-workflow-walkthrough)
   - [Implementation Reference](#implementation-reference)
2. [Data Preprocessing](#data-preprocessing)
   - [Overview](#overview-1)
   - [Preprocessing Workflow Walkthrough](#preprocessing-workflow-walkthrough)
     - [Location Parsing](#1-location-parsing)
     - [Text Cleaning](#2-text-cleaning)
     - [Sentence Splitting](#3-sentence-splitting)
   - [Implementation Reference](#implementation-reference-1)
3. [Classifier Modeling](#classifier-modeling)
   - [Overview](#overview-2)
   - [Modeling Workflow Walkthrough](#modeling-workflow-walkthrough)
   - [Key Observations](#key-observations)
   - [Conclusion](#conclusion)
4. [Classifier Modeling - LSTM Neural Network Classifier](#classifier-modeling---lstm-neural-network-classifier)
   - [Overview](#overview-3)
   - [Modeling Workflow Walkthrough](#modeling-workflow-walkthrough-1)
   - [Results and Observations](#results-and-observations)
   - [Model Comparison: Naive Bayes vs. LSTM](#model-comparison-naive-bayes-vs-lstm)
   - [Next Steps](#next-steps)
5. [Classifier Modeling - BERT Classifier](#classifier-modeling---bert-classifier)
   - [Overview](#overview-4)
   - [Modeling Workflow Walkthrough](#modeling-workflow-walkthrough-2)
   - [Key Observations](#key-observations-1)
   - [Conclusion](#conclusion-1)
6. [Text Analysis - Bigram Analysis](#text-analysis---bigram-analysis)
   - [Methodology](#methodology)
   - [Text Analysis Workflow](#text-analysis-workflow)
     - [Text Cleansing](#1-text-cleansing)
     - [Bigram Analysis](#2-bigram-analysis)
     - [Categorization](#3-categorization)
   - [Conclusion](#conclusion-2)

---
# Data Collection - Scraping

The data scraping step in the Tech Mastery Encyclopedia project automates the collection of job descriptions from LinkedIn using Python, Selenium, and BeautifulSoup. This step is essential for accumulating comprehensive job data, which will be processed further in the project pipeline.

---

## Overview

The scraping process is structured to maximize efficiency and handle potential challenges that come with dynamic web content. Here's a high-level walkthrough of the flow:

---

### Scraping Workflow Walkthrough

1. **Automated Login**: 
   - Uses Selenium to automate logging into LinkedIn with a user-provided username and password.
   - Ensures access to job listings that require authentication.

2. **Job Search**:
   - Navigates to LinkedIn’s job search page and initiates a search for roles like "Data Science" in specific country.
   - Dynamically loads job listings for the specified search term.

3. **Extracting Job Listings**:
   - Identifies and extracts key job details, including titles, links, and locations from each listing on the page.
   - Handles variations and ensures data is captured even if some elements are missing.

4. **Pagination**:
   - Automates navigation through multiple pages of job results to collect a broader range of listings.
   - Clicks through pages seamlessly, ensuring comprehensive data extraction.

5. **Data Handling**:
   - Incorporates error handling to manage potential issues like timeouts or missing elements.
   - Uses scrolling techniques to ensure all jobs are loaded and accessible for extraction.

---

### Implementation Reference

To see the full code and detailed implementation of each step, refer to the notebook: **1.Linkedin_Scrapper.ipynb**.

---
# Data Preprocessing

In the preprocessing step, we prepare the raw job data for analysis by extracting structured location information and cleaning the job descriptions. This step ensures that our data is clean, standardized, and ready for modeling.

---

## Overview

The preprocessing workflow involves two main tasks:
1. **Location Parsing**: Categorizing and extracting structured location details from job postings.
2. **Text Cleaning**: Standardizing job descriptions for further analysis.

---

## Preprocessing Workflow Walkthrough

### 1. Location Parsing

We use a custom class, `JobLocationParser`, to extract and organize location data into city, state, country, and work type (e.g., On-site, Remote, Hybrid).

- **Initialization**: Sets up lists and dictionaries for US states, Canadian provinces, and supported countries.
- **Place of Work Extraction**: Identifies and removes work types from location strings.
- **State/Province Validation**: Checks if a location part is a valid US state or Canadian province.
- **Location Categorization**: Splits and categorizes location strings into structured data fields.
- **Data Processing**: Updates each job posting with structured location data.

Refer to **2.Linkedin_Loc_Extraction.ipynb** for the complete implementation.

---

### 2. Text Cleaning

We clean and standardize job descriptions to simplify text analysis and improve model performance.

- **HTML Tag Removal**: Uses BeautifulSoup to strip HTML tags from descriptions.
- **Whitespace Cleanup**: Removes unnecessary line breaks and extra spaces.
- **Lowercase Transformation**: Converts text to lowercase for uniformity.
- **Special Character Removal**: Cleans out unwanted characters while keeping essential punctuation.
- **Stopword Removal**: Uses NLTK to remove common stopwords.
- **Lemmatization**: Reduces words to their root form using WordNetLemmatizer.

The `clean_description(description)` function performs these steps, transforming job descriptions into clean, analyzable text. See **3.Linkedin_Preprocessing.ipynb** for code details.

---

### 3. Sentence Splitting

To prepare for manual tagging, we split job descriptions into individual sentences.

- **Sentence Tokenization**: Uses spaCy’s `en_core_web_sm` model to split text into sentences.
- **Handling New Lines and Hyphens**: Cleans and organizes sentences for better readability and tagging.

This approach facilitates manual annotation of sentences as either "Qualified" or "Description," aiding future model training.

---

## Implementation Reference

For full code and detailed explanations, refer to:
- **2.Linkedin_Loc_Extraction.ipynb** for location parsing.
- **3.Linkedin_Preprocessing.ipynb** for text cleaning and sentence splitting.

---

This preprocessing step is critical for transforming raw job data into a structured and clean format, setting the stage for effective analysis and modeling.

---

# Classifier Modeling

In the classifier modeling step, we use machine learning techniques to classify sentences from job descriptions as either "Qualification" or "Description." This involves several models and methods to ensure that we achieve robust and accurate performance.

---

## Overview

We implement multiple classification models to predict whether a sentence in a job description relates to specific qualifications. The models are trained on labeled text data, which has been carefully preprocessed and split into training and testing sets.

---

## Modeling Workflow Walkthrough

1. **Data Partitioning**:
   - We load the cleaned and labeled dataset, ensuring there are no missing values.
   - The data is split into training and testing sets, maintaining a fixed random seed for reproducibility.

2. **Text Vectorization**:
   - We convert text data into a numerical format using `CountVectorizer`, creating a bag-of-words model.
   - To preserve the importance of frequent terms (like "Python" or "Machine Learning"), we transform the text data without applying Inverse Document Frequency (IDF).

3. **Model Training and Evaluation**:
   - **Naive Bayes Classifier**: A simple yet effective baseline model known for its ability to handle text data. It demonstrated high performance and provided a reliable benchmark for further experimentation.
   - **Support Vector Machine (SVM)**: We tested SVMs with different kernels:
     - **Polynomial Kernel**: A complex model that underperformed, likely due to overfitting and high dimensionality challenges.
     - **Linear Kernel**: A simpler and more effective approach, performing comparably well given the nature of the data.
     - **RBF Kernel**: An SVM with non-linear flexibility that yielded decent results but did not surpass the linear kernel's performance.

4. **Model Comparisons**:
   - We evaluated models based on their accuracy and consistency, using cross-validation and visualizing results with box plots.
   - **Confusion Matrices**: Provided insights into the models' strengths and weaknesses by showing true positives, true negatives, false positives, and false negatives.
   - **ROC Curves and AUROC**: Assessed the models' ability to distinguish between the two classes, giving a more comprehensive view of their performance.

---

## Key Observations

- **Naive Bayes**: Emerged as a strong baseline model, demonstrating high accuracy and a balanced performance across classes.
- **Linear SVM**: Performed nearly as well as Naive Bayes, proving effective in high-dimensional text data scenarios.
- **RBF SVM**: Showed reasonable performance but didn't outperform the simpler linear model.
- **Polynomial SVM**: Struggled with classification accuracy, likely due to overfitting.

---

## Conclusion

From our analysis, Naive Bayes and Linear SVM stand out as the most suitable models for our task, combining high accuracy and reliable classification. While SVMs provided useful insights, the Naive Bayes model was selected as the primary classifier due to its simplicity and strong performance. Future steps will explore advanced models, such as LSTM Neural Networks, to potentially enhance our results.

For more detailed results and visualizations, refer to the notebook: **5.LinkedIn_ClassifierModelling.ipynb**.

---
# Classifier Modeling - LSTM Neural Network Classifier

In this step, we explore the use of a Long Short-Term Memory (LSTM) Neural Network to classify job description sentences into "Qualification" or "Description." The goal is to leverage the sequential nature of job descriptions and improve performance over traditional models like Naive Bayes.

## Overview

We utilize an LSTM-based model, built using TensorFlow and Keras, to process text data, aiming to capture more complex patterns and relationships. The approach includes data partitioning, tokenization, model architecture design, and training with techniques to handle class imbalance and mitigate overfitting.

---

## Modeling Workflow Walkthrough

1. **Data Partitioning**:
   - The cleaned and labeled dataset is split into training and testing sets, with a fixed random seed to ensure reproducibility.

2. **Tokenization and Padding**:
   - We use a tokenizer with a vocabulary limit and a maximum sequence length to convert text data into sequences of integers.
   - The sequences are padded to ensure uniform input size for the LSTM model.

3. **LSTM Model Architecture**:
   - **Embedding Layer**: Converts words into dense vectors for better representation.
   - **Stacked LSTM Layers**: Two LSTM layers are included, with regularization techniques to prevent overfitting.
   - **Dropout Layer**: Added to further reduce overfitting.
   - **Output Layer**: A Dense layer with sigmoid activation for binary classification.

4. **Model Compilation**:
   - An exponential decay schedule for the learning rate is applied, with the Adam optimizer selected for its efficiency.
   - The model is compiled with binary crossentropy loss and metrics like accuracy and recall to monitor performance.

5. **Training**:
   - **Class Weights**: Computed to balance the learning process, addressing class imbalance.
   - **Early Stopping**: Implemented to stop training when the model no longer improves on the validation set, ensuring the best weights are retained.
   - The model is trained over multiple epochs with validation and callbacks to optimize results.

---

## Results and Observations

- **Performance Metrics**:
  - The model demonstrated strong overall performance, with solid recall and precision for the "non-qualified" class and slightly more challenges in correctly identifying "qualified" sentences.
  - Training and validation metrics showed consistent improvement, although there was a slight indication of overfitting.

- **Confusion Matrix**:
  - The model showed better performance for one class compared to the other, revealing areas where further tuning or additional data might help.

- **Training Metrics**:
  - Recall and accuracy trends indicated steady progress, with the validation metrics stabilizing, demonstrating that the model generalizes well but has room for refinement.

---

## Model Comparison: Naive Bayes vs. LSTM

- **Naive Bayes Classifier**:
  - Performed well as a simple, reliable baseline model, effectively handling patterns in the text.
- **LSTM Classifier**:
  - Captured more complex relationships in the text, demonstrating the benefits of a deep learning approach, though with some challenges in balancing precision and recall across classes.

---

## Next Steps

To further enhance the model's performance, we will explore using a BERT model, which may offer better contextual understanding of job descriptions and improve classification accuracy.

For implementation details, refer to: **6.LinkedIn_LSTM_NN_Classifier_Modelling.ipynb**.

---

# Classifier Modeling - BERT Classifier

In this step, we fine-tune a pre-trained DistilBERT model for classifying job description sentences as either "Qualification" or "Description." BERT models excel at understanding the context of text, making them a powerful tool for this classification task.

---

## Overview

We use the `distilbert-base-uncased` model, a compact and efficient version of BERT, for sequence classification. The process involves configuring model parameters, tokenizing text, and setting up the training environment to leverage BERT’s contextual strengths.

---

## Modeling Workflow Walkthrough

1. **Setting Parameters**:
   - **Maximum Sequence Length**: Set to 100 tokens, ensuring that input texts are consistently sized, either padded or truncated to this length.
   - **Pre-trained Model**: `distilbert-base-uncased` is selected for its efficiency and capability to handle large-scale text data.

2. **Tokenization**:
   - We use `DistilBertTokenizer` to transform text into tokens that the model can process.
   - Parameters:
     - `max_length=100`: Limits the input length to 100 tokens.
     - `padding='max_length'`: Pads shorter texts to 100 tokens.
     - `truncation=True`: Truncates texts longer than 100 tokens.
     - `clean_up_tokenization_spaces=False`: Ensures spaces are preserved in a meaningful way.
   - Special tokens, like `[PAD]`, are explicitly defined for uniformity.

3. **Data Preparation**:
   - The labeled dataset is loaded, and we split it into training and testing sets using an 80/20 split with a fixed `random_state=42` to ensure reproducibility.
   - The text data is then tokenized and converted into TensorFlow-compatible formats using `from_tensor_slices`.

4. **Model Configuration**:
   - **DistilBERT Configuration**: Set `num_labels=2` for binary classification.
   - **Output Settings**: `output_hidden_states=False` to simplify the model by focusing on the final output layer.

5. **Model Building**:
   - We load `TFDistilBertForSequenceClassification` and configure it for our binary classification task.
   - **Model Architecture**:
     - Embedding and transformer layers from DistilBERT handle input sequences.
     - A final dense layer with a sigmoid activation function outputs the binary classification result.

6. **Model Compilation**:
   - **Optimizer**: Adam optimizer with a learning rate of `3e-5` to ensure stable and efficient training.
   - **Loss Function**: `SparseCategoricalCrossentropy` with `from_logits=True`, appropriate for binary classification tasks with logits output.
   - **Metrics**: We track `accuracy` to measure model performance during training.

7. **Training the Model**:
   - **Shuffling**: The training dataset is shuffled with a buffer size of 1000 to randomize the order of samples before each epoch.
   - **Batching**: Both the training and validation datasets are batched with a size of 32 for efficient processing.
   - **Epochs**: The model is trained for 3 epochs, providing multiple passes over the data to learn from it.
   - **Validation**: The test dataset is used for validation, helping to monitor and adjust model performance during training.

8. **Making Predictions**:
   - We use a custom function to tokenize input text, pass it through the model, and apply softmax to get class probabilities.
   - The class with the highest probability is selected as the final prediction using `argmax`.

---

## Key Observations

- **Contextual Understanding**: By leveraging DistilBERT, the model can effectively capture and utilize the contextual meaning of words within job descriptions.
- **Parameter Selection**: Using a learning rate of `3e-5` and 3 training epochs strikes a balance between learning efficiency and model stability.
- **Data Handling**: Tokenization parameters, such as maximum length and padding, are crucial for ensuring consistent input size, which is vital for BERT models.

---

## Conclusion

The BERT-based model is well-prepared to handle the complexity of job description texts, thanks to its powerful language understanding capabilities. Moving forward, we plan to deploy this model in production, as it provides a robust and efficient solution for classifying job descriptions.

For full implementation details, check: **6.LinkedIn_BERT_Classifier_Modelling.ipynb**.


---
# Classifier Modeling

In the classifier modeling step, we use machine learning techniques to classify sentences from job descriptions as either "Qualification" or "Description." This involves several models and methods to ensure that we achieve robust and accurate performance.

---

## Overview

We implement multiple classification models to predict whether a sentence in a job description relates to specific qualifications. The models are trained on labeled text data, which has been carefully preprocessed and split into training and testing sets.

---

## Modeling Workflow Walkthrough

1. **Data Partitioning**:
   - We load the cleaned and labeled dataset, ensuring there are no missing values.
   - The data is split into training and testing sets, maintaining a fixed random seed for reproducibility.

2. **Text Vectorization**:
   - We convert text data into a numerical format using `CountVectorizer`, creating a bag-of-words model.
   - To preserve the importance of frequent terms (like "Python" or "Machine Learning"), we transform the text data without applying Inverse Document Frequency (IDF).

3. **Model Training and Evaluation**:
   - **Naive Bayes Classifier**: A simple yet effective baseline model known for its ability to handle text data. It demonstrated high performance and provided a reliable benchmark for further experimentation.
   - **Support Vector Machine (SVM)**: We tested SVMs with different kernels:
     - **Polynomial Kernel**: A complex model that underperformed, likely due to overfitting and high dimensionality challenges.
     - **Linear Kernel**: A simpler and more effective approach, performing comparably well given the nature of the data.
     - **RBF Kernel**: An SVM with non-linear flexibility that yielded decent results but did not surpass the linear kernel's performance.

4. **Model Comparisons**:
   - We evaluated models based on their accuracy and consistency, using cross-validation and visualizing results with box plots.
   - **Confusion Matrices**: Provided insights into the models' strengths and weaknesses by showing true positives, true negatives, false positives, and false negatives.
   - **ROC Curves and AUROC**: Assessed the models' ability to distinguish between the two classes, giving a more comprehensive view of their performance.

---

## Key Observations

- **Naive Bayes**: Emerged as a strong baseline model, demonstrating high accuracy and a balanced performance across classes.
- **Linear SVM**: Performed nearly as well as Naive Bayes, proving effective in high-dimensional text data scenarios.
- **RBF SVM**: Showed reasonable performance but didn't outperform the simpler linear model.
- **Polynomial SVM**: Struggled with classification accuracy, likely due to overfitting.

---

## Conclusion

From our analysis, Naive Bayes and Linear SVM stand out as the most suitable models for our task, combining high accuracy and reliable classification. While SVMs provided useful insights, the Naive Bayes model was selected as the primary classifier due to its simplicity and strong performance. Future steps will explore advanced models, such as LSTM Neural Networks, to potentially enhance our results.

For more detailed results and visualizations, refer to the notebook: **5.LinkedIn_ClassifierModelling.ipynb**.

---
# Classifier Modeling - LSTM Neural Network Classifier

In this step, we explore the use of a Long Short-Term Memory (LSTM) Neural Network to classify job description sentences into "Qualification" or "Description." The goal is to leverage the sequential nature of job descriptions and improve performance over traditional models like Naive Bayes.

## Overview

We utilize an LSTM-based model, built using TensorFlow and Keras, to process text data, aiming to capture more complex patterns and relationships. The approach includes data partitioning, tokenization, model architecture design, and training with techniques to handle class imbalance and mitigate overfitting.

---

## Modeling Workflow Walkthrough

1. **Data Partitioning**:
   - The cleaned and labeled dataset is split into training and testing sets, with a fixed random seed to ensure reproducibility.

2. **Tokenization and Padding**:
   - We use a tokenizer with a vocabulary limit and a maximum sequence length to convert text data into sequences of integers.
   - The sequences are padded to ensure uniform input size for the LSTM model.

3. **LSTM Model Architecture**:
   - **Embedding Layer**: Converts words into dense vectors for better representation.
   - **Stacked LSTM Layers**: Two LSTM layers are included, with regularization techniques to prevent overfitting.
   - **Dropout Layer**: Added to further reduce overfitting.
   - **Output Layer**: A Dense layer with sigmoid activation for binary classification.

4. **Model Compilation**:
   - An exponential decay schedule for the learning rate is applied, with the Adam optimizer selected for its efficiency.
   - The model is compiled with binary crossentropy loss and metrics like accuracy and recall to monitor performance.

5. **Training**:
   - **Class Weights**: Computed to balance the learning process, addressing class imbalance.
   - **Early Stopping**: Implemented to stop training when the model no longer improves on the validation set, ensuring the best weights are retained.
   - The model is trained over multiple epochs with validation and callbacks to optimize results.

---

## Results and Observations

- **Performance Metrics**:
  - The model demonstrated strong overall performance, with solid recall and precision for the "non-qualified" class and slightly more challenges in correctly identifying "qualified" sentences.
  - Training and validation metrics showed consistent improvement, although there was a slight indication of overfitting.

- **Confusion Matrix**:
  - The model showed better performance for one class compared to the other, revealing areas where further tuning or additional data might help.

- **Training Metrics**:
  - Recall and accuracy trends indicated steady progress, with the validation metrics stabilizing, demonstrating that the model generalizes well but has room for refinement.

---

## Model Comparison: Naive Bayes vs. LSTM

- **Naive Bayes Classifier**:
  - Performed well as a simple, reliable baseline model, effectively handling patterns in the text.
- **LSTM Classifier**:
  - Captured more complex relationships in the text, demonstrating the benefits of a deep learning approach, though with some challenges in balancing precision and recall across classes.

---

## Next Steps

To further enhance the model's performance, we will explore using a BERT model, which may offer better contextual understanding of job descriptions and improve classification accuracy.

For implementation details, refer to: **6.LinkedIn_LSTM_NN_Classifier_Modelling.ipynb**.

---

# Classifier Modeling - BERT Classifier

In this step, we fine-tune a pre-trained DistilBERT model for classifying job description sentences as either "Qualification" or "Description." BERT models excel at understanding the context of text, making them a powerful tool for this classification task.

---

## Overview

We use the `distilbert-base-uncased` model, a compact and efficient version of BERT, for sequence classification. The process involves configuring model parameters, tokenizing text, and setting up the training environment to leverage BERT’s contextual strengths.

---

## Modeling Workflow Walkthrough

1. **Setting Parameters**:
   - **Maximum Sequence Length**: Set to 100 tokens, ensuring that input texts are consistently sized, either padded or truncated to this length.
   - **Pre-trained Model**: `distilbert-base-uncased` is selected for its efficiency and capability to handle large-scale text data.

2. **Tokenization**:
   - We use `DistilBertTokenizer` to transform text into tokens that the model can process.
   - Parameters:
     - `max_length=100`: Limits the input length to 100 tokens.
     - `padding='max_length'`: Pads shorter texts to 100 tokens.
     - `truncation=True`: Truncates texts longer than 100 tokens.
     - `clean_up_tokenization_spaces=False`: Ensures spaces are preserved in a meaningful way.
   - Special tokens, like `[PAD]`, are explicitly defined for uniformity.

3. **Data Preparation**:
   - The labeled dataset is loaded, and we split it into training and testing sets using an 80/20 split with a fixed `random_state=42` to ensure reproducibility.
   - The text data is then tokenized and converted into TensorFlow-compatible formats using `from_tensor_slices`.

4. **Model Configuration**:
   - **DistilBERT Configuration**: Set `num_labels=2` for binary classification.
   - **Output Settings**: `output_hidden_states=False` to simplify the model by focusing on the final output layer.

5. **Model Building**:
   - We load `TFDistilBertForSequenceClassification` and configure it for our binary classification task.
   - **Model Architecture**:
     - Embedding and transformer layers from DistilBERT handle input sequences.
     - A final dense layer with a sigmoid activation function outputs the binary classification result.

6. **Model Compilation**:
   - **Optimizer**: Adam optimizer with a learning rate of `3e-5` to ensure stable and efficient training.
   - **Loss Function**: `SparseCategoricalCrossentropy` with `from_logits=True`, appropriate for binary classification tasks with logits output.
   - **Metrics**: We track `accuracy` to measure model performance during training.

7. **Training the Model**:
   - **Shuffling**: The training dataset is shuffled with a buffer size of 1000 to randomize the order of samples before each epoch.
   - **Batching**: Both the training and validation datasets are batched with a size of 32 for efficient processing.
   - **Epochs**: The model is trained for 3 epochs, providing multiple passes over the data to learn from it.
   - **Validation**: The test dataset is used for validation, helping to monitor and adjust model performance during training.

8. **Making Predictions**:
   - We use a custom function to tokenize input text, pass it through the model, and apply softmax to get class probabilities.
   - The class with the highest probability is selected as the final prediction using `argmax`.

---

## Key Observations

- **Contextual Understanding**: By leveraging DistilBERT, the model can effectively capture and utilize the contextual meaning of words within job descriptions.
- **Parameter Selection**: Using a learning rate of `3e-5` and 3 training epochs strikes a balance between learning efficiency and model stability.
- **Data Handling**: Tokenization parameters, such as maximum length and padding, are crucial for ensuring consistent input size, which is vital for BERT models.

---

## Conclusion

The BERT-based model is well-prepared to handle the complexity of job description texts, thanks to its powerful language understanding capabilities. Moving forward, we plan to deploy this model in production, as it provides a robust and efficient solution for classifying job descriptions.

For full implementation details, check: **6.LinkedIn_BERT_Classifier_Modelling.ipynb**.

---

# Text - Bigram Analysis

The text analysis step in the project processes job descriptions to identify and categorize relevant keywords into predefined categories such as tools, libraries, languages, skills, and education. This step leverages natural language processing (NLP) techniques, regular expressions, and MongoDB to automate and streamline data analysis.

---

## Methodology

Our approach uses Python’s NLTK library for bigram analysis and custom regex patterns for keyword matching. By combining data cleansing and keyword categorization, we extract meaningful insights from job descriptions stored in MongoDB.

---

### Text Analysis Workflow

#### 1. Text Cleansing

The initial step involves cleaning and preparing text data:
- **Punctuation Removal**: All punctuation marks, such as commas, periods, and special characters, are removed to simplify the text.
- **Stopword Removal**: Common English stopwords are filtered out using NLTK, ensuring that only meaningful words are retained.
- **Lowercasing**: Text is converted to lowercase to maintain consistency across the dataset.

---

#### 2. Bigram Analysis

We use bigram analysis to identify common word pairs:
- **Corpus Preparation**: The text data is combined into a single corpus after cleansing.
- **Bigram Extraction**: NLTK’s `BigramCollocationFinder` is used to extract word pairs, applying a frequency filter to discard infrequent bigrams.
- **Frequency Calculation**: Frequencies of each bigram are computed, and only those meeting a specified threshold are considered for further analysis.

---

#### 3. Categorization

Bigrams are categorized into one of the predefined groups:
- **Category Matching**: Using regular expressions, bigrams are matched to categories such as:
  - **Tools**
  - **Libraries**
  - **Languages**
  - **Skills**
  - **Education**
- **Mixed Category**: Bigrams that do not fit into any specific category are labeled as "mixed" and may require additional review.


---

## Conclusion

This text analysis methodology effectively processes large volumes of job descriptions, extracting and categorizing bigrams to provide structured insights. By employing text cleansing, bigram analysis, and keyword matching, our approach enables accurate data extraction, setting the stage for further modeling and analysis in the Tech Mastery Encyclopedia project.

For a full walkthrough of the implementation, please refer to the provided Jupyter notebooks.
