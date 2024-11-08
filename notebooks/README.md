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

