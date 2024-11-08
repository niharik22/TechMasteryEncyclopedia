# Tech Mastery Encyclopedia

Tech Mastery Encyclopedia is a comprehensive project aimed at gathering, processing, and analyzing open-source job description data from various sources. Inspired by **SkillQuery**, the goal is to develop a centralized repository of job-related information to help uncover key trends and skills in the tech industry.

---

## Project Overview

Tech Mastery Encyclopedia aims to extract, clean, and model job data to identify and classify skills and qualifications across a variety of tech roles. The project involves multiple stages, including data collection, cleansing, and applying natural language processing models for text classification.

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
