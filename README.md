# Sri Lanka Paddy Production - Streamlit Dashboard

A Streamlit companion to the Applied Machine Learning notebook.

## Windows setup

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

`train_models.py` trains all five models on the included CSV, saves the complete
preprocessing + model pipelines to `models/`, and saves test metrics and predictions.
It may take a few minutes, particularly for the RF + ANN model.

The notebook in this folder is the companion experimental record; the Streamlit
dashboard is a separate Python app. If running the notebook in Colab, its model
files will not automatically appear on your Windows computer: run the training
script locally as shown above.

## Method

Inputs: District, Season, Year, Sown_Extent_Hectares.
Target: Production_Thousand_Metric_Tons.

Random 80/20 train/test split, random_state=42. The same split is used for
Random Forest, XGBoost, LightGBM, CatBoost and RF + ANN.

The test split is **not chronological**, so scores do not establish accuracy
for future-year forecasting. Predictions are educational scenarios.

Source: https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics
Verify this is the exact DCS page used to obtain the original files.
