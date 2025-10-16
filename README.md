# Obesity Prediction Project

A data analysis project that explores obesity levels based on lifestyle factors, demographics, and eating habits.

## Project Overview

This project analyzes obesity data to understand what factors contribute to different obesity levels. We explore the dataset, clean the data, and visualize patterns to gain insights.

The dataset classifies individuals into 7 obesity categories:
- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

## Dataset

The dataset contains information about 2,111 individuals with features including:

**Demographics:**
- Gender (Male/Female)
- Age (years)
- Height (meters)
- Weight (kilograms)

**Health & Habits:**
- Family history of obesity
- Smoking habits
- High calorie food consumption (FAVC)
- Vegetable consumption frequency (FCVC)
- Number of main meals (NCP)
- Food between meals (CAEC)
- Water consumption (CH2O)
- Calorie monitoring (SCC)
- Alcohol consumption (CALC)

**Lifestyle:**
- Physical activity frequency (FAF)
- Technology usage time (TUE)
- Transportation method (MTRANS)

## Project Structure

```
obesity_project/
├── data/
│   ├── obesity_prediction.csv        # Original dataset
│   ├── obesity_cleaned.csv            # Cleaned dataset
│   ├── obesity_numeric_cleaned.csv    # Encoded numeric dataset
│   └── encoding_mappings.json         # Encoding reference
├── notebooks/
│   └── 01_data_exploration.ipynb      # Data exploration & cleaning
├── requirements.txt                    # Dependencies
└── README.md                          # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab or Jupyter Notebook

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd obesity_project
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open `notebooks/01_data_exploration.ipynb`

3. Run all cells to see the complete analysis

## What We Did

### Data Cleaning
- Checked for missing values (none found!)
- Removed 24 duplicate rows (2111 → 2087 records)
- Verified data ranges for age, height, weight
- Identified and analyzed outliers
- Added BMI calculation (Weight / Height²)

### Data Encoding
We converted categorical variables to numbers for analysis:
- Binary features (yes/no) → 0/1
- Frequency features (no/sometimes/frequently/always) → 0/1/2/3
- Transportation types → 0-4
- Obesity levels → 0-6

### Visualizations
- Obesity level distribution
- Age distribution
- Height vs Weight scatter plot
- BMI distribution
- Family history impact
- Physical activity patterns
- Correlation heatmap

## Key Findings

- Dataset has no missing values
- Found and removed 24 duplicate rows
- Final dataset: 2,087 unique records
- BMI is highly correlated with obesity level (as expected)
- Family history appears to have an impact on obesity
- Age distribution is mostly young adults
- Weight is strongly related to obesity levels

## Dependencies

```
pandas
numpy
matplotlib
seaborn
jupyterlab
```

See `requirements.txt` for full list with versions.

## Team

This is a collaborative project for data analysis and exploration.

## Future Work

- Build machine learning models to predict obesity levels
- Create interactive visualizations
- Analyze feature importance
- Test different ML algorithms

---

**Note:** This is a student project for learning data analysis and visualization.
