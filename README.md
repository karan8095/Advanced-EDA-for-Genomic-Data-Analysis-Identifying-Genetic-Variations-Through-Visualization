# Genomic Data Analysis App

## Overview
This Streamlit-based web application enables users to upload genomic datasets (CSV files) and perform exploratory data analysis (EDA), data visualization, and machine learning classification using a Random Forest model. If no dataset is provided, the app generates synthetic genomic data for demonstration purposes.

## Features
- Upload a CSV file containing genomic data
- View dataset statistics and class distribution
- Visualize SNP variant distribution
- Perform Principal Component Analysis (PCA) and t-SNE for dimensionality reduction
- Train and evaluate a Random Forest Classifier

## Technologies Used
- **Python**
- **Streamlit** (for web application development)
- **NumPy, Pandas** (for data manipulation)
- **Matplotlib, Seaborn** (for data visualization)
- **Scikit-learn** (for machine learning and dimensionality reduction)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/genomic-data-analysis.git
   cd genomic-data-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload a CSV file with genomic data, or use the generated synthetic data.
3. Explore the dataset using the interactive interface.
4. View visualizations of SNP distributions and dimensionality reduction techniques.
5. Train a Random Forest model and evaluate its performance.

## Dataset Format
The uploaded CSV file should have the following format:
- Columns representing SNP features (e.g., `SNP_0`, `SNP_1`, ... `SNP_n`)
- A `Target` column indicating class labels (binary classification: 0 or 1)

Example:
```
SNP_0,SNP_1,SNP_2,...,SNP_1499,Target
0,2,1,...,0,1
1,0,2,...,2,0
...
```

## Sample Synthetic Data
If no dataset is uploaded, the app generates a synthetic dataset with 4,000 samples and 1,500 SNP features.

## Future Improvements
- Implement support for multi-class classification
- Add additional machine learning models for comparison
- Enhance visualization options with interactive plots

## License
This project is licensed under the MIT License.

## Author
Karan B



