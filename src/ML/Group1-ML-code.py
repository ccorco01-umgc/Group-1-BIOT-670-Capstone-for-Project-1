# First test of machine learning model
# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 2. Load your Excel data
file_path = "test.xlsx" 
df = pd.read_excel(file_path)

# Drop datetime columns if any
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        print(f"Dropping datetime column: {col}")
        df = df.drop(columns=[col])

# 3. Define features and target
target_col = "Organism"
categorical_columns = ["Wind Direction", "Rain Y/N", "Climate Type", "Season", "Rural/Urban"]

# Ensure the RH% column exists
if "RH%" not in df.columns:
    raise ValueError("Expected column 'RH%' not found in Excel file!")

X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Encode categorical columns
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 5. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Optional: Save model and encoders for later use
joblib.dump(model, "rf_model.pkl")
joblib.dump(le_dict, "le_dict.pkl")
joblib.dump(le_target, "le_target.pkl")

# 7. Function for user input prediction
def predict_organism(model, le_dict, categorical_columns, le_target):
    import pandas as pd

    # Get user input
    temp = float(input("Enter temperature (°C): "))
    wind_speed = float(input("Enter wind speed (MPH): "))
    wind_dir = input("Enter wind direction (N, S, E, W): ")
    elevation = float(input("Enter elevation (m): "))
    humidity = float(input("Enter relative humidity (%): "))
    rain = input("Was there rain? (Y/N): ")
    climate = input("Enter climate type (Coastal, Grassland, Tundra, Forest, Wetland, Desert): ")
    season = input("Enter season: ")
    rural_urban = input("Rural, Suburban, Urban?: ")

    # Put into DataFrame
    data = {
        "Temp. In C": [temp],
        "Wind Speed (MPH)": [wind_speed],
        "Wind Direction": [wind_dir],
        "Elevation (m)": [elevation],
        "RH%": [humidity],
        "Rain Y/N": [rain],
        "Climate Type": [climate],
        "Season": [season],
        "Rural/Urban": [rural_urban]
    }
    df_input = pd.DataFrame(data)

    # Encode categorical columns
    for col in categorical_columns:
        df_input[col] = le_dict[col].transform(df_input[col].astype(str))

    # Reorder columns to match training data
    df_input = df_input[X_train.columns]

    # Predict
    pred = model.predict(df_input)
    pred_label = le_target.inverse_transform(pred)
    print("Predicted organism:", pred_label[0])


predict_organism(model, le_dict, categorical_columns, le_target)