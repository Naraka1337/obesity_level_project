# 🏥 Obesity Prediction Project

A comprehensive machine learning project that predicts obesity levels based on lifestyle factors, demographic information, and eating habits. This project uses supervised learning to classify individuals into different obesity categories.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Description](#-dataset-description)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📁 File Descriptions](#-file-descriptions)
- [🔧 Data Processing](#-data-processing)
- [📈 Data Exploration](#-data-exploration)
- [🤖 Machine Learning Pipeline](#-machine-learning-pipeline)
- [📊 Results and Visualizations](#-results-and-visualizations)
- [🌐 Future Features](#-future-features)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

This project aims to predict obesity levels using machine learning algorithms based on various lifestyle and demographic factors. The model can classify individuals into 7 different obesity categories:

1. **Insufficient Weight** (0)
2. **Normal Weight** (1)
3. **Overweight Level I** (2)
4. **Overweight Level II** (3)
5. **Obesity Type I** (4)
6. **Obesity Type II** (5)
7. **Obesity Type III** (6)

### 🎯 Key Features

- **Comprehensive Data Processing**: Automated conversion of categorical data to numeric format
- **Data Visualization**: Interactive plots and statistical analysis
- **Machine Learning Models**: Multiple algorithms for obesity prediction
- **Feature Engineering**: BMI calculation and feature optimization
- **Data Quality Assurance**: Missing value detection and data validation

## 📊 Dataset Description

The dataset contains information about **2,113 individuals** with the following features:

### 👤 Demographic Information
- **Gender**: Male/Female
- **Age**: Age in years
- **Height**: Height in meters
- **Weight**: Weight in kilograms

### 🏥 Health History
- **Family History**: Family history of obesity (Yes/No)
- **Smoking**: Smoking habits (Yes/No)

### 🍽️ Eating Habits
- **FAVC**: Frequent consumption of high caloric food (Yes/No)
- **FCVC**: Frequency of consumption of vegetables (1-3 scale)
- **NCP**: Number of main meals (1-4 scale)
- **CAEC**: Consumption of food between meals (No/Sometimes/Frequently/Always)
- **CH2O**: Consumption of water daily (1-3 scale)
- **SCC**: Calories consumption monitoring (Yes/No)
- **CALC**: Consumption of alcohol (No/Sometimes/Frequently/Always)

### 🏃‍♀️ Physical Activity
- **FAF**: Physical activity frequency (0-3 scale)
- **TUE**: Time using technology devices (0-2 scale)

### 🚗 Transportation
- **MTRANS**: Transportation used (Public Transportation/Walking/Automobile/Motorbike/Bike)

### 🎯 Target Variable
- **Obesity**: Obesity level classification (7 categories)

## 🏗️ Project Structure

```
obesity_project/
├── 📁 data/                          # Dataset files
│   ├── obesity_ prediction.csv       # Raw dataset
│   ├── obesity_numeric.csv          # Processed numeric dataset
│   └── mappings.json                # Feature encoding mappings
├── 📁 src/                          # Source code
│   └── preprocess.py                # Data preprocessing script
├── 📁 notebooks/                    # Jupyter notebooks
│   └── 01_data_exploration.ipynb    # Data exploration notebook
├── 📁 models/                       # Trained models (future)
├── 📁 reports/                      # Generated reports (future)
├── 📁 venv/                         # Virtual environment
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd obesity_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Data Preprocessing**
   ```bash
   cd src
   python preprocess.py
   ```

2. **Data Exploration**
   ```bash
   jupyter lab
   # Open notebooks/01_data_exploration.ipynb
   ```

## 📁 File Descriptions

### 📊 Data Files

- **`obesity_ prediction.csv`**: Raw dataset with categorical variables
- **`obesity_numeric.csv`**: Processed dataset with all features converted to numeric format
- **`mappings.json`**: Dictionary mapping categorical values to numeric codes

### 🐍 Source Code

- **`src/preprocess.py`**: Main preprocessing script that:
  - Loads raw data
  - Converts categorical variables to numeric
  - Calculates BMI
  - Saves processed data and mappings

### 📓 Notebooks

- **`notebooks/01_data_exploration.ipynb`**: Comprehensive data analysis including:
  - Data loading and inspection
  - Statistical summaries
  - Categorical variable analysis
  - Data visualization
  - Correlation analysis

## 🔧 Data Processing

### Preprocessing Steps

1. **Data Loading**: Read raw CSV file
2. **Categorical Encoding**: Convert text to numbers
   - Binary variables: Yes/No → 1/0
   - Frequency variables: No/Sometimes/Frequently/Always → 0/1/2/3
   - Transportation: Different methods → 0-4
3. **BMI Calculation**: Weight / (Height)²
4. **Data Validation**: Check for missing values and inconsistencies
5. **Export**: Save processed data and mappings

### Feature Encoding

| Feature | Encoding |
|---------|----------|
| Gender | Female: 0, Male: 1 |
| Family History | No: 0, Yes: 1 |
| FAVC | No: 0, Yes: 1 |
| SMOKE | No: 0, Yes: 1 |
| SCC | No: 0, Yes: 1 |
| CAEC | No: 0, Sometimes: 1, Frequently: 2, Always: 3 |
| CALC | No: 0, Sometimes: 1, Frequently: 2, Always: 3 |
| MTRANS | Public: 0, Walking: 1, Car: 2, Motorbike: 3, Bike: 4 |
| Obesity | Insufficient: 0, Normal: 1, Overweight I: 2, Overweight II: 3, Obesity I: 4, Obesity II: 5, Obesity III: 6 |

## 📈 Data Exploration

The exploration notebook provides comprehensive analysis:

### 🔍 Data Quality Assessment
- Dataset shape and structure
- Data types and memory usage
- Missing value analysis
- Statistical summaries

### 📊 Visualizations
- **Obesity Distribution**: Count plot showing distribution across categories
- **BMI Analysis**: Histogram and box plots of BMI values
- **Correlation Analysis**: Heatmap of feature correlations
- **Lifestyle Factors**: Analysis of eating habits, physical activity, and transportation

### 🎯 Key Insights
- Balanced dataset across obesity categories
- Strong correlation between BMI and obesity levels
- Lifestyle factors show meaningful patterns
- No missing values in the dataset

## 🤖 Machine Learning Pipeline

### Planned Features

1. **Data Splitting**: Train/validation/test sets
2. **Feature Scaling**: Normalization and standardization
3. **Model Training**: Multiple algorithms
   - Random Forest
   - Support Vector Machine
   - Logistic Regression
   - Neural Networks
4. **Model Evaluation**: Cross-validation and metrics
5. **Hyperparameter Tuning**: Grid search optimization
6. **Model Persistence**: Save trained models

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curves

## 📊 Results and Visualizations

### Current Visualizations

- **Distribution Plots**: Obesity category distribution
- **BMI Analysis**: Distribution and relationship with obesity
- **Correlation Heatmaps**: Feature relationships
- **Box Plots**: Lifestyle factors by obesity level

### Future Visualizations

- Model performance comparisons
- Feature importance plots
- Prediction confidence intervals
- Interactive dashboards

## 🌐 Future Features

### 🔮 Planned Enhancements

1. **Web Application**
   - Interactive prediction interface using Streamlit
   - User-friendly input forms
   - Real-time predictions

2. **Model Deployment**
   - REST API for predictions
   - Docker containerization
   - Cloud deployment

3. **Advanced Analytics**
   - Feature importance analysis
   - SHAP values for interpretability
   - A/B testing framework

4. **Data Pipeline**
   - Automated data validation
   - Model retraining pipeline
   - Performance monitoring

5. **Reporting**
   - Automated report generation
   - Model performance dashboards
   - Business insights

## 🛠️ Dependencies

The project uses the following Python packages:

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Plotting and visualization
seaborn         # Statistical data visualization
scikit-learn    # Machine learning algorithms
joblib          # Model persistence
streamlit       # Web application framework
jupyterlab      # Interactive development environment
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset source and contributors
- Open source libraries and tools used
- Community feedback and suggestions

---

**⭐ If you found this project helpful, please give it a star!**
