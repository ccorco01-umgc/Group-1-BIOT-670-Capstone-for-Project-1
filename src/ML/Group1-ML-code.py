# Group 1 - ML Model for Airborne Microbial Communities
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
# Ensure you are in same directory as data and script
file_path = "ML-data.xlsx"
df = pd.read_excel(file_path)

# Cleaning columns
numeric_columns = [
    "Temperature (°C)",
    "Wind Speed (MPH)",
    "Altitude (km above sea level)",
    "Relative Humidity (%)",
    "Sea salt Concentration ug/m^3",
    "SO4 Concentration ug/m^3",
    "Organic Carbon Concentration ug/m^3",
    "Dust Concentration ug/m^3",
    "Black Carbon Concentration ug/m^3",
    "Concentration (cfu/m^3)",
    "Sampling duration"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(" ", ""), errors="coerce")

# Drop rows with missing numeric values
df.dropna(subset=[c for c in numeric_columns if c in df.columns], inplace=True)

# Drop extra columns
drop_cols = [
    "Sample ID", "DOI", "Date", "Continent", "Country", "City/Area",
    "Kingdom", "Phylum", "Class", "Family", "Genus", "Species",
    "Gene Amp. Method", "Extraction Method", "Gene Promoter",
    "Primer", "Sequencing Platform", "Airflow speed"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# Clean columns
df = df.dropna(subset=["Organism", "% Abundance", "Allergen (Y/N)"])

# Column encoding
binary_cols = ["Rain Y/N"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].replace({"Y": 1, "N": 0, "y": 1, "n": 0}).astype(int)

categorical_cols = ["Rural/Urban", "Climate Type", "Wind Direction"]
le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

le_target = LabelEncoder()
df["Organism_encoded"] = le_target.fit_transform(df["Organism"])

# Inputs and targets
features = [
    "Rural/Urban", "Climate Type", "Wind Speed (MPH)", "Wind Direction",
    "Rain Y/N", "Relative Humidity (%)", "Sampling duration",
    "Sea salt Concentration ug/m^3", "SO4 Concentration ug/m^3",
    "Organic Carbon Concentration ug/m^3", "Dust Concentration ug/m^3",
    "Black Carbon Concentration ug/m^3", "Altitude (km above sea level)",
    "Temperature (°C)", "Concentration (cfu/m^3)"
]

X = df[[c for c in features if c in df.columns]]
y_organism = df["Organism_encoded"]
y_allergen = df["Allergen (Y/N)"]
y_abundance = df["% Abundance"]

# Data split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_organism, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_allergen, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_abundance, test_size=0.2, random_state=42)

# Training
clf_organism = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf_organism.fit(X_train_c, y_train_c)


clf_allergen = RandomForestClassifier(n_estimators=200, random_state=42)
clf_allergen.fit(X_train_a, y_train_a)

reg_abundance = RandomForestRegressor(n_estimators=200, random_state=42)
reg_abundance.fit(X_train_r, y_train_r)


# Prediction
def predict():
    user_input = {}

    prompts = {
        "Rural/Urban": "Enter area type (Rural, Suburban, Urban): ",
        "Climate Type": "Enter climate type (Mountain, Grassland, Forest, Desert, Tundra, Coastal, Wetland): ",
        "Wind Speed (MPH)": "Enter wind speed in MPH: ",
        "Wind Direction": "Enter wind direction (N, S, E, W): ",
        "Rain Y/N": "Is it raining? (Y/N): ",
        "Relative Humidity (%)": "Enter relative humidity (%): ",
        "Temperature (°C)": "Enter temperature (°C): ",
        "Sampling duration": "Enter sampling duration (hours): ",
        "Sea salt Concentration ug/m^3": "Enter sea salt concentration (ug/m^3, leave blank for 0): ",
        "SO4 Concentration ug/m^3": "Enter SO4 concentration (ug/m^3, leave blank for 0): ",
        "Organic Carbon Concentration ug/m^3": "Enter organic carbon concentration (ug/m^3, leave blank for 0): ",
        "Dust Concentration ug/m^3": "Enter dust concentration (ug/m^3, leave blank for 0): ",
        "Black Carbon Concentration ug/m^3": "Enter black carbon concentration (ug/m^3, leave blank for 0): ",
        "Altitude (km above sea level)": "Enter altitude (km above sea level, leave blank for 0): ",
        "Concentration (cfu/m^3)": "Enter concentration (cfu/m^3, leave blank for 0): "
    }

    for col in features:
        if col == "Allergen (Y/N)":
            continue
        prompt = prompts.get(col, f"Enter {col}: ")
        val = input(prompt).strip()
        if val == "":
            val = 0 
        user_input[col] = val

    df_input = pd.DataFrame([user_input])

    numeric_only = [c for c in features if c not in categorical_cols + ["Rain Y/N", "Allergen (Y/N)"]]
    for col in numeric_only:
        if col in df_input.columns:
            try:
                df_input[col] = df_input[col].astype(float)
            except:
                df_input[col] = 0

    if "Rain Y/N" in df_input.columns:
        df_input["Rain Y/N"] = df_input["Rain Y/N"].replace({"Y": 1, "N": 0, "y": 1, "n": 0}).astype(int)

    for col in categorical_cols:
        if col in df_input.columns and col in le_dict:
            val = df_input[col].astype(str)
            df_input[col] = df_input[col].apply(lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else 0)

    for col in X.columns:
        if col not in df_input:
            df_input[col] = 0

    df_input = df_input[X.columns]

    org_pred = clf_organism.predict(df_input)
    org_label = le_target.inverse_transform(org_pred)

    allergen_pred = clf_allergen.predict(df_input)
    allergen_label = "Yes" if allergen_pred[0] == 1 else "No"

    abundance_pred = reg_abundance.predict(df_input)

    print(f"\nPredicted Organism: {org_label[0]}")
    print(f"Predicted Allergen: {allergen_label}")
    print(f"Predicted % Abundance: {abundance_pred[0]:.2f}%")


# Run
predict()
