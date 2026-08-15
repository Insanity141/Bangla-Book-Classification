# Bangla Book Category Classification

Statistics for Data Science mini-project on multi-label classification of Bangla books using book titles, summaries, and selected metadata.

## Project Information

**Course:** Statistics for Data Science  
**Project:** Project 4 — Book Category Classification from Title and Summary  
**Semester:** Summer 2026

---

## 1. Project Overview

Each book in the dataset can belong to one or more categories. Therefore, the problem is naturally formulated as a **multi-label classification task**.

The objective of this project is to investigate how well book categories can be predicted using:

- Book title
- Book summary
- Author information
- Publisher information

Three different models are developed and compared using **Precision, Recall, and F1-score**.

---

## 2. Research Question

The main research question is:

> **How effectively can frequent Bangla book categories be predicted from book titles, summaries, and related metadata using a classical multi-label classification approach?**

The project also investigates:

1. Does cleaning the scraped book-summary text improve classification performance?
2. Does adding author and publisher metadata improve classification performance?

---

## 3. Dataset

The project uses the Bangla book dataset provided for the CSE303 mini-project.

The main data sources are:

- `book.json` — book-level metadata
- `category.json` — category information
- `book_to_category.json` — book-to-category relationships
- `author.json` — author metadata
- `publisher.json` — publisher metadata

### Initial Dataset Statistics

The raw book table contains:

- **149,515** book records
- **127,302** unique book IDs
- **235,475** book-category relationships
- **1,493** categories actually represented in the relationship data
- **107,680** unique books with at least one category

### Category Selection

Exploratory analysis showed a strong long-tail distribution of category frequencies. Many categories contain very few books, making them difficult to learn reliably and increasing the computational cost of the multi-label classification problem.

We therefore selected categories having:

> **At least 100 books**

This retained:

- **478 categories**
- **102,227 books**

Books whose category assignments contained none of the selected 478 categories were excluded from the modeling population.

The selection rule was based on the observed category-frequency distribution.

---

## 4. Exploratory Data Analysis

The EDA focused on:

- Dataset structure and missing values
- Duplicate book IDs and exact duplicate records
- Multi-label category structure
- Category-frequency imbalance
- Category coverage
- Title quality and duplication
- Summary length and scraping artifacts
- Author frequency
- Publisher frequency
- Metadata completeness

### Multi-label Structure

Among books with category assignments:

- Average categories per book: **2.19**
- Maximum categories per book: **30**
- Books with multiple categories: **62,120**
- Multi-label books: **57.69%**

These results support formulating the task as a multi-label classification problem.

### Category Imbalance

The category distribution is strongly right-skewed.

- Median category size: **58 books**
- Maximum category size: **1,826 books**
- Categories with fewer than 5 books: **62**
- Categories with fewer than 10 books: **158**
- Categories with fewer than 20 books: **312**

This long-tail distribution motivated the use of the minimum-frequency category-selection rule.

### Summary Quality

The raw summary field showed substantial variation:

- Mean summary length: **424.58 characters**
- Median: **9 characters**
- Maximum: **64,106 characters**

A major scraping artifact was the occurrence of `"Show More"` in summaries.

The EDA found:

- **81,779** exact `"Show More"` summaries
- **149,173** summaries containing `"Show More"` somewhere
- **8** summaries containing HTML tags

The `"Show More"` artifact and HTML tags were removed during the preprocessing used for Models 2 and 3.

### Title Quality

The title field was highly complete:

- Empty titles: **0**
- Mean title length: **22.95 characters**
- Median title length: **20 characters**
- Maximum title length: **155 characters**

Duplicate titles were common, but repeated titles were not assumed to represent duplicate books. Exact duplicate rows were investigated separately.

### Author Metadata

- Unique authors appearing in books: **44,846**
- Median books per author: **1**
- Authors with exactly one book: **29,516**
- Authors with at least 10 books: **2,382**

Author IDs therefore show a strong long-tail distribution.

### Publisher Metadata

- Unique publishers appearing in books: **6,045**
- Median books per publisher: **3**
- Publishers with exactly one book: **2,064**
- Publishers with at least 10 books: **1,691**

Publisher IDs are also long-tailed, so rare values were grouped for modeling.

### Metadata Completeness

Within the final modeling population of **102,227 books**:

- Missing `author_id`: **0**
- Missing `publisher_id`: **27**

Author information was therefore complete, while publisher information was available for almost all modeling records.

---

## 5. Modeling Approach

All three models use:

> **One-vs-Rest Logistic Regression**

The models differ mainly in the information supplied to the classifier.

### Model 1 — Baseline

**Input:**

- Raw book title
- Raw book summary

**Feature representation:**

- TF-IDF
- Up to 30,000 features
- Unigrams and bigrams
- `min_df = 2`

**Classifier:**

- One-vs-Rest Logistic Regression

Model 1 provides a baseline using minimally processed textual information.

### Model 2 — Cleaned Text

Model 2 uses the same books, categories, train-validation-test split, TF-IDF configuration, and classifier as Model 1.

The main difference is preprocessing:

- Remove `"Show More"` scraping artifacts
- Remove HTML tags
- Normalize whitespace
- Handle missing summaries

The purpose of Model 2 is to determine whether cleaning the scraped text improves classification performance.

### Model 3 — Cleaned Text + Metadata

Model 3 extends Model 2 by adding selected metadata.

**Inputs:**

- Cleaned book title
- Cleaned book summary
- Frequently occurring author IDs
- Frequently occurring publisher IDs

For metadata:

- Author IDs appearing at least 10 times were retained as explicit author categories.
- Publisher IDs appearing at least 10 times were retained as explicit publisher categories.
- Rare author and publisher values were grouped.
- Missing publisher information was represented using an unknown category.

The categorical metadata was one-hot encoded and combined with the TF-IDF text features.

---

## 6. Train–Validation–Test Setup

The final modeling population was divided into:

- **70% training**
- **15% validation**
- **15% test**

The same split was reused across all three models to make the comparison fair.

TF-IDF was fitted using the training text only and then applied to the validation and test sets.

Metadata encoders were also fitted using training data only.

---

## 7. Evaluation Metrics

Because the task is multi-label classification, model performance was evaluated using:

- **Precision**
- **Recall**
- **F1-score**

Weighted averaging was used for the reported metrics.

---

## 8. Results

### Model Comparison

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Model 1 — Raw Text | 0.4044 | 0.0612 | 0.0974 |
| Model 2 — Cleaned Text | 0.3902 | 0.0595 | 0.0947 |
| **Model 3 — Cleaned Text + Metadata** | **0.6518** | **0.1491** | **0.2226** |

### Interpretation

Model 1 establishes the baseline using raw title and summary text. Its low Recall indicates that the baseline model predicts only a small portion of the true category assignments.

Model 2 does not improve upon Model 1. Precision, Recall, and F1-score decrease slightly, indicating that the basic text-cleaning steps used in this experiment were not sufficient to improve classification performance.

Model 3 produces the strongest result. Adding selected author and publisher metadata substantially improves Precision, Recall, and F1-score compared with both text-only models.

The improvement from Model 2 to Model 3 suggests that author and publisher information provides useful predictive information beyond cleaned title and summary text alone.

However, the final Recall remains relatively low, indicating that the multi-label classification problem is still challenging and that the models miss many of the true category assignments.

---

## 9. Project Figures

The report-ready figures generated during the analysis are stored in:

```text
results/figures/

```

## 10. Trained Model Artifacts

The trained Logistic Regression classifier files are excluded from the repository because of their file size.

The complete training and evaluation pipeline is provided in:

`notebooks/02_model_development.ipynb`

Running the corresponding training cells will regenerate Model 1, Model 2, and Model 3. The notebook also contains the code used to save, load, predict with, and evaluate the trained models.

Smaller supporting artifacts, such as TF-IDF vectorizers and metadata encoders, are included in `results/models/`.