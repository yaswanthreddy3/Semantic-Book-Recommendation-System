# Semantic-Book-Recommendation-System
AI-powered Semantic Book Recommendation System that suggests books based on user queries using semantic similarity, emotion, and category filters. Built with Python, Flask, LangChain, HuggingFace embeddings, and Chroma, it provides personalized, relevant, and interactive book recommendations.

## Overview
This repository contains a Flask web application that provides AI-powered semantic book recommendations. The system allows users to search for books based on keywords, emotions, and categories, leveraging embeddings and vector similarity for personalized suggestions.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Dependencies](#setup-and-dependencies)
3. [Data Overview](#data-overview)
4. [Embeddings and Chroma DB](#embeddings-and-chroma-db)
5. [Recommendation Engine](#recommendation-engine)
6. [How to Use](#how-to-use)
7. [Deployment](#deployment)

## Introduction
The goal of this project is to suggest relevant books using semantic similarity of descriptions and additional filters:
- Handles user queries intelligently
- Filters by book category and emotional tone (Happy, Sad, Angry, Surprising, Suspenseful)
- Prioritizes exact title matches for better user experience
- Provides book details such as title, authors, rating, description, and thumbnail

## Setup and Dependencies
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `flask`
- `langchain`
- `langchain-chroma`
- `huggingface-hub`
- `sentence-transformers`

Install dependencies using:
```bash
pip install -r requirements.txt

Data Overview

books_with_emotion_scores.csv – Main dataset with book metadata, emotion scores, and ratings.

tagged_description.txt – Text file containing book descriptions for semantic indexing.

book_db/ – Local Chroma vector database directory (generated automatically).

Embeddings and Chroma DB

Uses HuggingFaceEmbeddings (MiniLM model) to convert book descriptions into vector embeddings.

Stores embeddings in a Chroma vector database for fast semantic search.

Automatically builds or loads the database on app startup.

Recommendation Engine

Semantic search on book descriptions using vector similarity.

Exact title matching is prioritized for better results.

Filters by category and tone.

Returns top recommendations sorted by relevance or rating.

Supports paginated display in the frontend.

How to Use

Clone the repository:

git clone https://github.com/your-username/semantic-book-recommendation.git


Navigate to the project directory:

cd semantic-book-recommendation


Place books_with_emotion_scores.csv and tagged_description.txt in the project root.

Run the Flask app:

python app.py


Open a browser at http://localhost:5001 and enter a book name, keyword, or query.

Deployment

Can be deployed on Render, Heroku, or any server supporting Python & Flask.

Ensure CSV and Chroma database folder are included for correct functionality.

License

This project is licensed under the MIT License. See the LICENSE file for details.


I can also **prepare a ready-to-upload `.md` file** for you with this content so you just copy and upload to GitHub.  

Do you want me to do that?

