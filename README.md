
# Streamlit NLP Dashboard

Welcome to the **Streamlit NLP Dashboard** project! This dashboard provides a user-friendly interface for performing natural language processing (NLP) tasks, including sentiment analysis, word clouds, and more, all powered by a streamlined Python backend. The application is built with **Streamlit** for easy deployment and interaction.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview
The **Streamlit NLP Dashboard** is designed to allow users to quickly visualize and analyze textual data with various NLP techniques. The dashboard supports multiple functionalities, including generating word clouds, performing sentiment analysis, and visualizing data insights.

## Features
- **Sentiment Analysis**: Analyze the sentiment of text data and visualize the results.
- **Word Cloud Generation**: Create word clouds from text data to visualize word frequency.
- **Word Frequency**: Display the most common words in the text data.
- **Text Clustering**: Cluster similar text data together for better organization.
- **Real-Time Analysis**: Input custom text for instant analysis and visualization.
- **Download Results**: Download the analysis results for further exploration.

## Installation
Follow these steps to get the project running on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DarrelBumika/NLP-Dashboard.git
   cd NLP-Dashboard
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run App.py
   ```

## Usage
After launching the app, you can navigate to the dashboard via your browser. The main page includes options to:
- Upload a CSV or text file for analysis.
- Select between different NLP analysis tools, including sentiment analysis and word cloud generation.
- Custom text input for real-time analysis.
- Download the analysis results for further exploration.

## Project Structure
```
NLP-Dashboard/
│
├── .streamlit/               # Streamlit configuration settings
│   └── config.toml
│
├── crawling/                 # Twitter scraping scripts for data collection
│   ├── DataCleaning.ipynb
│   ├── DataCrawling.ipynb 
│   ├── PredictSentiment.ipynb
│   └── TextClustering.ipynb
│
├── data/                     # Sample datasets for testing
│   ├── cleaned-data.csv  
│   └── raw-data.csv
│
├── models/                   # Pre-trained models for sentiment analysis
│   ├── feature-bow.p
│   └── model-nb.p
|
├── pages/                    # Additional pages for the Streamlit app
│   └── Dashboard.py
│
├── src/                      # Custom CSS styles and images
│   └── style.css
│
├── pages/
│   └── page1.py              # Additional pages for the Streamlit app
│
├── utils/
│   └── DataProcessing.py     # Utility functions for NLP tasks
│
├── App.py                    # Main Streamlit app entry point
├── README.md                 # Project overview and instructions
└── requirements.txt          # List of dependencies
```
