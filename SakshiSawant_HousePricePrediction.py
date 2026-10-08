"""
House Price Prediction using Machine Learning
IBM SkillsBuild - Data Analytics with AI | Academic Internship Program (BharatCares x AICTE)

Objective : Analyse a residential property dataset and build a regression model that
            predicts the price of a house (in Rs lakh).
Workflow  : Load data -> Clean -> EDA -> Feature engineering -> Model building
            -> Tuning -> Evaluation -> Save model -> Insights

Run       : python YourName_HousePricePrediction.py
Output    : charts in ./figures, trained model in ./house_price_model.pkl
"""


# ======================================================================
# 1. Import Libraries
# ======================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (9, 5)
RANDOM_STATE = 42
print(os.makedirs("figures", exist_ok=True))


# ======================================================================
# 2. Load the Dataset
# ======================================================================
DATA_PATH = "house_prices.csv"

def generate_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    location = rng.choice(["Central", "Premium", "Suburban", "Outskirts"], n, p=[0.25, 0.15, 0.40, 0.20])
    loc_base = {"Premium": 1.45, "Central": 1.20, "Suburban": 1.0, "Outskirts": 0.78}
    dist_range = {"Premium": (1, 8), "Central": (1, 10), "Suburban": (6, 22), "Outskirts": (15, 40)}

    area = np.clip(rng.normal(1250, 420, n), 350, 3500).round()
    bedrooms = np.clip((area / 450 + rng.normal(0, 0.6, n)).round(), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n) + rng.integers(0, 2, n), 1, 6)
    age = rng.integers(0, 36, n)
    total_floors = rng.integers(1, 26, n)
    floor = np.array([rng.integers(0, tf + 1) for tf in total_floors])
    parking = rng.choice([0, 1, 2], n, p=[0.25, 0.55, 0.20])
    furnishing = rng.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n, p=[0.35, 0.40, 0.25])
    lift = ((total_floors > 4) & (rng.random(n) > 0.1)).astype(int)
    metro = (rng.random(n) < 0.3).astype(int)
    distance = np.array([rng.uniform(*dist_range[l]) for l in location]).round(1)

    price = (area * 0.055                      # base: ~5,500 per sq ft (lakh)
             + bedrooms * 3.5 + bathrooms * 2.5
             + parking * 4.0 + lift * 3.0 + metro * 6.0
             + pd.Series(furnishing).map({"Unfurnished": 0, "Semi-Furnished": 4, "Furnished": 9}).values
             - age * 0.55 - distance * 0.9
             + np.minimum(floor, 15) * 0.25)
    price = price * pd.Series(location).map(loc_base).values
    price = price * rng.lognormal(0, 0.07, n) + 8
    price = np.clip(price, 12, None).round(2)

    df = pd.DataFrame({
        "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "property_age_years": age, "floor": floor, "total_floors": total_floors,
        "parking_spaces": parking, "location": location, "furnishing": furnishing,
        "has_lift": lift, "near_metro": metro, "distance_to_center_km": distance,
        "price_lakh": price,
    })
    # Make the data messy like real life: missing values, duplicates, a few outliers
    for col, frac in [("property_age_years", 0.03), ("distance_to_center_km", 0.02), ("furnishing", 0.02), ("bathrooms", 0.015)]:
        df.loc[df.sample(frac=frac, random_state=seed).index, col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)
    out_idx = df.sample(8, random_state=seed + 1).index
    df.loc[out_idx, "price_lakh"] = df.loc[out_idx, "price_lakh"] * 4
    return df

if not os.path.exists(DATA_PATH):
    generate_dataset().to_csv(DATA_PATH, index=False)
    print("Dataset created:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())


# ======================================================================
# 3. Understand the Data
# ======================================================================
df.info()
print()

print(df.describe().T.round(2))

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print("\nDuplicate rows:", df.duplicated().sum())


# ======================================================================
# 4. Data Cleaning
# ======================================================================
df_clean = df.drop_duplicates().copy()
print("Rows after removing duplicates:", len(df_clean))

num_cols = df_clean.select_dtypes(include="number").columns.drop("price_lakh")
cat_cols = df_clean.select_dtypes(exclude="number").columns

for c in num_cols:
    df_clean[c] = df_clean[c].fillna(df_clean[c].median())
for c in cat_cols:
    df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])

# bathrooms / age should be whole numbers
df_clean[["bathrooms", "property_age_years"]] = df_clean[["bathrooms", "property_age_years"]].round().astype(int)

print("Missing values left:", df_clean.isnull().sum().sum())

# Outlier detection on the target using the IQR rule
q1, q3 = df_clean["price_lakh"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 3 * iqr          # 3*IQR keeps genuine luxury homes, removes data errors
lower = max(q1 - 3 * iqr, 0)
outliers = df_clean[(df_clean["price_lakh"] > upper) | (df_clean["price_lakh"] < lower)]
print(f"Price outliers found: {len(outliers)}  (valid range: {lower:.1f} - {upper:.1f} lakh)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=df_clean["price_lakh"], ax=ax[0]).set_title("Price – before outlier removal")
df_clean = df_clean[(df_clean["price_lakh"] >= lower) & (df_clean["price_lakh"] <= upper)]
sns.boxplot(x=df_clean["price_lakh"], ax=ax[1]).set_title("Price – after outlier removal")
plt.tight_layout(); plt.savefig("figures/01_outliers.png", dpi=150); plt.show()
print("Final clean dataset shape:", df_clean.shape)


# ======================================================================
# 5. Exploratory Data Analysis (EDA)
# ======================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_clean["price_lakh"], bins=40, kde=True, ax=ax[0]).set_title("House Price Distribution")
sns.histplot(np.log1p(df_clean["price_lakh"]), bins=40, kde=True, ax=ax[1], color="green").set_title("Log-transformed Price")
ax[0].set_xlabel("Price (₹ lakh)"); ax[1].set_xlabel("log(1 + Price)")
plt.tight_layout(); plt.savefig("figures/02_price_distribution.png", dpi=150); plt.show()
print("Skewness of price:", round(df_clean["price_lakh"].skew(), 2))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
order = df_clean.groupby("location")["price_lakh"].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x="location", y="price_lakh", order=order, ax=ax[0]).set_title("Price by Location")
sns.barplot(data=df_clean, x="furnishing", y="price_lakh", estimator=np.mean, ax=ax[1],
            order=["Unfurnished", "Semi-Furnished", "Furnished"]).set_title("Average Price by Furnishing")
for a in ax: a.set_ylabel("Price (₹ lakh)")
plt.tight_layout(); plt.savefig("figures/03_price_by_location.png", dpi=150); plt.show()

print(df_clean.groupby("location")["price_lakh"].agg(["count", "mean", "median"]).round(1).sort_values("mean", ascending=False))

plt.figure(figsize=(9, 5.5))
sns.scatterplot(data=df_clean, x="area_sqft", y="price_lakh", hue="location", alpha=0.6)
plt.title("Area vs Price (coloured by location)")
plt.xlabel("Area (sq ft)"); plt.ylabel("Price (₹ lakh)")
plt.tight_layout(); plt.savefig("figures/04_area_vs_price.png", dpi=150); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.boxplot(data=df_clean, x="bedrooms", y="price_lakh", ax=ax[0]).set_title("Price by Bedrooms")
sns.regplot(data=df_clean, x="property_age_years", y="price_lakh", scatter_kws={"alpha": .3, "s": 12}, line_kws={"color": "red"}, ax=ax[1]).set_title("Price vs Property Age")
sns.regplot(data=df_clean, x="distance_to_center_km", y="price_lakh", scatter_kws={"alpha": .3, "s": 12}, line_kws={"color": "red"}, ax=ax[2]).set_title("Price vs Distance to Centre")
plt.tight_layout(); plt.savefig("figures/05_bedrooms_age_distance.png", dpi=150); plt.show()

corr = df_clean.select_dtypes(include="number").corr()
plt.figure(figsize=(10, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.5)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("figures/06_correlation.png", dpi=150); plt.show()

print("Correlation of each feature with price:")
print(corr["price_lakh"].drop("price_lakh").sort_values(ascending=False).round(3))


# ======================================================================
# 6. Feature Engineering
# ======================================================================
df_fe = df_clean.copy()
df_fe["floor_ratio"] = (df_fe["floor"] / df_fe["total_floors"]).round(3)
df_fe["area_per_bedroom"] = (df_fe["area_sqft"] / df_fe["bedrooms"]).round(1)
print(df_fe[["floor", "total_floors", "floor_ratio", "area_sqft", "bedrooms", "area_per_bedroom"]].head())


# ======================================================================
# 7. Train / Test Split and Preprocessing Pipeline
# ======================================================================
X = df_fe.drop(columns="price_lakh")
y = df_fe["price_lakh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "| Test:", X_test.shape)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])
print("Numeric features    :", numeric_features)
print("Categorical features:", categorical_features)


# ======================================================================
# 8. Model Building & Comparison
# ======================================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = []
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "CV R²": cv_r2,
        "Test R²": r2_score(y_test, pred),
        "MAE (lakh)": mean_absolute_error(y_test, pred),
        "RMSE (lakh)": np.sqrt(mean_squared_error(y_test, pred)),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False).round(4).reset_index(drop=True)
print(results_df)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=results_df, x="Test R²", y="Model", ax=ax[0]).set_title("Test R² (higher is better)")
sns.barplot(data=results_df, x="MAE (lakh)", y="Model", ax=ax[1], color="salmon").set_title("MAE in ₹ lakh (lower is better)")
ax[0].set_xlim(0, 1)
plt.tight_layout(); plt.savefig("figures/07_model_comparison.png", dpi=150); plt.show()


# ======================================================================
# 9. Hyper-parameter Tuning (Gradient Boosting)
# ======================================================================
param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.7, 0.85, 1.0],
}
base = Pipeline([("prep", preprocessor), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
search = RandomizedSearchCV(base, param_dist, n_iter=12, cv=5, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best parameters:", search.best_params_)
print("Best CV R²     :", round(search.best_score_, 4))


# ======================================================================
# 10. Final Model Evaluation
# ======================================================================
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f} lakh")
print(f"RMSE     : {rmse:.2f} lakh")
print(f"MAPE     : {mape:.2f} %")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(y_test, y_pred, alpha=0.5, s=18)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax[0].plot(lims, lims, "r--"); ax[0].set_xlabel("Actual Price (₹ lakh)"); ax[0].set_ylabel("Predicted Price (₹ lakh)")
ax[0].set_title("Actual vs Predicted")

residuals = y_test - y_pred
sns.histplot(residuals, bins=35, kde=True, ax=ax[1], color="purple")
ax[1].set_title("Residual Distribution"); ax[1].set_xlabel("Residual (₹ lakh)")
plt.tight_layout(); plt.savefig("figures/08_actual_vs_predicted.png", dpi=150); plt.show()


# ======================================================================
# 11. Feature Importance – What drives house prices?
# ======================================================================
feature_names = best_model.named_steps["prep"].get_feature_names_out()
importances = pd.Series(best_model.named_steps["model"].feature_importances_, index=feature_names)
importances.index = importances.index.str.replace("num__", "").str.replace("cat__", "")
top = importances.sort_values(ascending=False).head(12)

plt.figure(figsize=(9, 5.5))
sns.barplot(x=top.values, y=top.index, color="teal")
plt.title("Top 12 Feature Importances"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("figures/09_feature_importance.png", dpi=150); plt.show()
print(top.round(4))


# ======================================================================
# 12. Save the Model & Predict a New House
# ======================================================================
joblib.dump(best_model, "house_price_model.pkl")

new_house = pd.DataFrame([{
    "area_sqft": 1400, "bedrooms": 3, "bathrooms": 2, "property_age_years": 5,
    "floor": 7, "total_floors": 15, "parking_spaces": 1, "location": "Central",
    "furnishing": "Semi-Furnished", "has_lift": 1, "near_metro": 1, "distance_to_center_km": 4.5,
}])
new_house["floor_ratio"] = new_house["floor"] / new_house["total_floors"]
new_house["area_per_bedroom"] = new_house["area_sqft"] / new_house["bedrooms"]

loaded = joblib.load("house_price_model.pkl")
print(f"Predicted price: Rs. {loaded.predict(new_house)[0]:.2f} lakh")


# ======================================================================
# 13. Key Insights & Conclusion
# ======================================================================