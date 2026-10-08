# 🏠 House Price Prediction using Machine Learning

**Program:** IBM SkillsBuild – Data Analytics with AI Academic Internship Program
(conducted by BharatCares in association with AICTE)

## 📌 Project Description
This project analyses a residential property dataset and builds a machine learning model that predicts the
**price of a house (in ₹ lakh)** from features such as area, bedrooms, bathrooms, property age, floor,
parking, location, furnishing, lift, metro access and distance from the city centre.

The project covers the complete data-analytics workflow: data loading, cleaning, exploratory data analysis (EDA),
feature engineering, model training, hyper-parameter tuning, evaluation and insight generation.

## 📂 Dataset
* **File:** `house_prices.csv` (1,515 rows × 13 columns before cleaning)
* **Link:** `<PASTE YOUR GITHUB / GOOGLE DRIVE LINK TO house_prices.csv HERE>`
* **Note:** The dataset is a simulated housing dataset with realistic price relationships. It intentionally contains
  missing values, duplicate rows and outliers to demonstrate data cleaning. The notebook can regenerate it
  automatically if the file is missing, and can also be pointed at any real housing CSV (e.g. from Kaggle).

| Column | Description |
|---|---|
| area_sqft | Built-up area in square feet |
| bedrooms / bathrooms | Number of bedrooms / bathrooms |
| property_age_years | Age of the property |
| floor / total_floors | Floor of the flat / floors in the building |
| parking_spaces | Number of parking spaces |
| location | Central, Premium, Suburban or Outskirts |
| furnishing | Unfurnished, Semi-Furnished or Furnished |
| has_lift / near_metro | 1 = yes, 0 = no |
| distance_to_center_km | Distance from the city centre |
| **price_lakh** | **Target – price in ₹ lakh** |

## 🛠️ Technologies Used
Python 3 · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter Notebook

## 🔬 Methodology
1. **Data cleaning** – removed 15 duplicates, imputed missing values (median / mode), removed 7 extreme price outliers (3×IQR rule)
2. **EDA** – price distribution, price by location/furnishing, area vs price, correlation heatmap
3. **Feature engineering** – `floor_ratio`, `area_per_bedroom`
4. **Modelling** – Linear Regression, Ridge, Random Forest, Gradient Boosting (5-fold cross-validation)
5. **Tuning** – RandomizedSearchCV on Gradient Boosting
6. **Evaluation** – R², MAE, RMSE, MAPE on a held-out 20 % test set

## 📊 Results
| Model | Test R² | MAE (₹ lakh) | RMSE (₹ lakh) |
|---|---|---|---|
| Gradient Boosting (before tuning) | 0.9536 | 6.51 | 8.51 |
| Linear Regression | 0.9478 | 6.83 | 9.02 |
| Ridge Regression | 0.9476 | 6.84 | 9.04 |
| Random Forest | 0.9332 | 7.91 | 10.21 |
| **Gradient Boosting (tuned) – final** | **0.9579** | **6.07** | **8.11** |

**Key insights:** area is the strongest price driver, followed by distance from the city centre and location tier;
older properties are cheaper; amenities add a smaller premium.

## ▶️ How to Run
```bash
# 1. Clone / download this repository and open a terminal in the folder
# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook YourName_HousePricePrediction.ipynb
```
Then choose **Kernel → Restart & Run All**. The notebook saves charts in `figures/` and the trained model as
`house_price_model.pkl`. It also runs on Google Colab – upload the notebook (and optionally `house_prices.csv`) and run all cells.

## 📁 Project Structure
```
├── YourName_HousePricePrediction.ipynb   # Main code
├── house_prices.csv                      # Dataset
├── requirements.txt                      # Dependencies
├── YourName_ProjectReport.docx           # Project report
└── README.md                             # This file
```

## 🚀 Future Scope
Use real listing data, add neighbourhood features (schools, hospitals), try XGBoost / LightGBM,
and deploy the model as a Streamlit web app.
