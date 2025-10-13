import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_csv(path):
    return pd.read_csv(path)

def simple_preprocess(df, target_col="taxon_A_abundance"):
    numeric = df.select_dtypes(include="number")
    if target_col not in numeric.columns:
        raise ValueError(f"Target column {target_col} not found")
    X = numeric.drop(columns=[target_col])
    y = numeric[target_col]
    imp = SimpleImputer(strategy="median")
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(imp.fit_transform(X), columns=X.columns),
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test
