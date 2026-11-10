# Indian House Price Prediction (Python + ML)

## What this prototype does
- Predicts an approximate Indian residential property price in INR.
- Takes city, locality, BHK, built-up area, city-centre/outskirts, property age, furnishing, parking and gated-community inputs.
- Compares locality-level average prices per square foot.
- Displays a 2020–2026 illustrative trend and 2027–2030 projection.
- Uses a scikit-learn Linear Regression pipeline.

## Important dataset warning
Both CSV files included here are **fabricated synthetic educational data** generated for this prototype. They are not actual listings, registered sale transactions, or verified market prices. Historical and future charts are illustrative only. Do not use the predictions for a real purchase, loan, valuation or investment decision.

The demo contains 15 cities and multiple localities per city. The model is useful for demonstrating the end-to-end ML workflow, not for claiming real-world accuracy.

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP folder.
3. Open Command Prompt in the extracted folder.
4. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501.

## Files
- `app.py` — Streamlit application and ML pipeline.
- `indian_house_prices_synthetic.csv` — generated sample property records.
- `locality_price_trends_synthetic.csv` — illustrative annual locality price-per-square-foot trends.
- `requirements.txt` — Python dependencies.

## How the ML part works
1. Separate input features from `Price_INR`.
2. One-hot encode city, locality and furnishing.
3. Standardize numerical features.
4. Split records into 80% training and 20% testing.
5. Train Linear Regression.
6. Evaluate using MAE and R-squared.
7. Predict the price for user-entered property details.

## Replacing demo data with real data
For credible future work:
- NHB RESIDEX: https://residex.nhbonline.org.in/ — official city housing price indices and trends.
- RBI DBIE House Price Index: https://dbie.rbihub.in/statistics?table=real_sector.house_price_index_rn — quarterly city-level house price index data.
- Kaggle Indian Urban House Price Prediction: https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data — educational synthetic listing dataset with property-level fields.

An index is not the same as an individual property's sale price. For accurate locality-level prediction, obtain verified property transaction/listing data with consistent fields, dates, area definitions and sale prices. Check dataset license and permitted use before incorporating it.

## Academic limitations / next improvements
- Add a real property-level dataset and document its source.
- Avoid random train/test leakage if multiple records represent the same project or locality/time period; use a time-based or group-based split where appropriate.
- Compare Linear Regression with Random Forest and Gradient Boosting.
- Report MAE, RMSE and R-squared on a genuine held-out dataset.
- Consider log-price modelling and outlier checks.
- Treat future projections as uncertain scenarios, not guaranteed prices.
