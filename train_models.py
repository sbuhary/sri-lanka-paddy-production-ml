from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from model_wrappers import CompatibleCatBoost

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Sri_Lanka_Paddy_Production.csv"
OUT = BASE / "models"
OUT.mkdir(exist_ok=True)
FEATURES = ["District", "Season", "Year", "Sown_Extent_Hectares"]
TARGET = "Production_Thousand_Metric_Tons"

data = pd.read_csv(DATA)
paddy = data[FEATURES + [TARGET]].dropna(subset=["Sown_Extent_Hectares", TARGET]).copy()
X, y = paddy[FEATURES], paddy[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

prep = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     ["District", "Season"])
], remainder="passthrough")
rf = Pipeline([("prepare", clone(prep)),
               ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
xgb = Pipeline([("prepare", clone(prep)),
                ("model", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=.05,
                                     objective="reg:squarederror", random_state=42, n_jobs=-1))])
lgbm = Pipeline([("prepare", clone(prep)),
                 ("model", LGBMRegressor(n_estimators=300, learning_rate=.05,
                                       max_depth=6, verbosity=-1, random_state=42))])
cat = Pipeline([("prepare", clone(prep)),
                ("model", CompatibleCatBoost(iterations=300, learning_rate=.05,
                                           depth=6, random_seed=42))])
nn_prep = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     ["District", "Season"]),
    ("numbers", StandardScaler(), ["Year", "Sown_Extent_Hectares"])
])
nn = Pipeline([("prepare", nn_prep),
               ("model", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1200,
                                      early_stopping=True, random_state=42))])
hybrid = VotingRegressor([("forest", clone(rf)), ("neural_net", nn)])
models = {
    "Random Forest": rf, "XGBoost": xgb, "LightGBM": lgbm,
    "CatBoost": cat, "RF + ANN": hybrid
}
filenames = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "CatBoost": "catboost.joblib",
    "RF + ANN": "rf_ann.joblib",
}
rows = []
for name, model in models.items():
    print("Training", name, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "R2": float(r2_score(y_test, pred)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred)))
    })
    joblib.dump(model, OUT / filenames[name])
    print(f"  R2={rows[-1]['R2']:.4f}, RMSE={rows[-1]['RMSE']:.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame({"Actual": y_test.to_numpy(), **{
    name: model.predict(X_test) for name, model in models.items()
}}).to_csv(OUT / "test_predictions.csv", index=False)
(OUT / "metadata.json").write_text(json.dumps({
    "models": filenames,
    "districts": sorted(paddy.District.dropna().unique().tolist()),
    "seasons": sorted(paddy.Season.dropna().unique().tolist()),
    "year_min": int(paddy.Year.min()),
    "year_max": int(paddy.Year.max()),
    "rows": int(len(paddy)),
    "original_rows": int(len(data)),
    "split": "Random 80/20, random_state=42",
    "target": TARGET
}, indent=2), encoding="utf-8")
print("Saved trained pipelines and evaluation data to models/", flush=True)
