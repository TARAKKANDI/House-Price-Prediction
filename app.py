import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Indian House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Indian House Price Prediction")
st.caption("A student prototype using Python, Pandas, Matplotlib and Linear Regression")

DATA_PATH = "indian_house_prices_synthetic.csv"
TREND_PATH = "locality_price_trends_synthetic.csv"
df = pd.read_csv(DATA_PATH)
trends = pd.read_csv(TREND_PATH)

FEATURES = ["City","Locality","BHK","Area_sqft","City_Centre","Property_Age_Years","Furnishing","Parking","Gated_Community"]
TARGET = "Price_INR"
cat_cols = ["City","Locality","Furnishing"]
num_cols = [c for c in FEATURES if c not in cat_cols]
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numeric", StandardScaler(), num_cols)
])
model = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)

def inr(value):
    value = max(0, int(round(value)))
    s = str(value)
    if len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        parts=[]
        while len(rest)>2:
            parts.insert(0,rest[-2:])
            rest=rest[:-2]
        if rest: parts.insert(0,rest)
        s=",".join(parts)+","+last3
    return "₹" + s

with st.sidebar:
    st.header("Project information")
    st.write("**Model:** Linear Regression")
    st.write(f"**Training records:** {len(df):,}")
    st.write(f"**Cities:** {df.City.nunique()}")
    st.write("**Dataset:** Fabricated demo data")
    st.info("This is an academic prototype. Do not use its estimates for an actual property purchase.")

tab1, tab2, tab3, tab4 = st.tabs(["🏡 Price Prediction","📊 Locality Comparison","📈 Past & Future Trends","ℹ️ Model & Dataset"])

with tab1:
    st.subheader("Enter your house requirements")
    c1,c2,c3 = st.columns(3)
    city = c1.selectbox("City", sorted(df.City.unique()), index=0)
    localities = sorted(df.loc[df.City==city,"Locality"].unique())
    locality = c2.selectbox("Locality / Area", localities)
    bhk = c3.selectbox("Bedrooms", [1,2,3,4], index=1, format_func=lambda x:f"{x} BHK")
    c4,c5,c6 = st.columns(3)
    area = c4.number_input("Built-up area (sq. ft.)", min_value=300, max_value=5000, value=1000, step=50)
    city_centre = c5.selectbox("Location type", ["Outskirts","Near / within city centre"])
    age = c6.slider("Property age (years)", 0, 40, 5)
    c7,c8,c9 = st.columns(3)
    furnishing = c7.selectbox("Furnishing", ["Unfurnished","Semi-furnished","Furnished"])
    parking = c8.selectbox("Parking available?", ["Yes","No"])
    gated = c9.selectbox("Gated community?", ["Yes","No"])
    if st.button("Predict House Price", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "City":city,"Locality":locality,"BHK":bhk,"Area_sqft":area,
            "City_Centre":1 if city_centre=="Near / within city centre" else 0,
            "Property_Age_Years":age,"Furnishing":furnishing,
            "Parking":1 if parking=="Yes" else 0,
            "Gated_Community":1 if gated=="Yes" else 0
        }])
        estimate = float(model.predict(row)[0])
        low, high = estimate*.90, estimate*1.10
        st.success("Prediction completed")
        a,b,c = st.columns(3)
        a.metric("Estimated house price", inr(estimate))
        b.metric("Approx. lower range", inr(low))
        c.metric("Approx. upper range", inr(high))
        st.caption("The ±10% range is a simple illustrative band, not a statistically calibrated confidence interval.")
        st.write(f"Estimated rate: **{inr(estimate/area)} per sq. ft.**")
        st.write("**Note:** The model learns patterns from the included synthetic demo dataset. Results are not verified market valuations.")

with tab2:
    st.subheader("Compare localities within a city")
    selected_city = st.selectbox("Choose city for comparison", sorted(df.City.unique()), key="compare_city")
    compare = df[df.City==selected_city].groupby("Locality",as_index=False).agg(
        Avg_Price_per_sqft=("Price_INR", lambda x: np.nan),
        Avg_Total_Price=("Price_INR","mean")
    )
    # Compute rate from individual records
    rate_df = df[df.City==selected_city].assign(rate=lambda x:x.Price_INR/x.Area_sqft).groupby("Locality",as_index=False).rate.mean()
    compare = rate_df.rename(columns={"rate":"Avg_Price_per_sqft"}).merge(compare[["Locality","Avg_Total_Price"]],on="Locality")
    compare = compare.sort_values("Avg_Price_per_sqft")
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.barh(compare.Locality, compare.Avg_Price_per_sqft)
    ax.set_xlabel("Average estimated price per sq. ft. (₹)")
    ax.set_title(f"Locality price comparison — {selected_city} (synthetic demo data)")
    ax.grid(axis="x", alpha=.25)
    st.pyplot(fig)
    st.dataframe(compare.assign(Avg_Price_per_sqft=compare.Avg_Price_per_sqft.round(0).astype(int),
                                Avg_Total_Price=compare.Avg_Total_Price.round(0).astype(int)), use_container_width=True)
    st.caption("Lower bars indicate relatively cheaper localities in this demo dataset, not necessarily the cheapest real-world options.")

with tab3:
    st.subheader("Historical trend and future estimate")
    city_t = st.selectbox("Select city", sorted(trends.City.unique()), key="trend_city")
    locs_t = sorted(trends[trends.City==city_t].Locality.unique())
    loc_t = st.selectbox("Select locality", locs_t, key="trend_locality")
    t = trends[(trends.City==city_t)&(trends.Locality==loc_t)].copy()
    hist_t = t[t.Year<=2026]
    future_t = t[t.Year>=2026]
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.plot(hist_t.Year, hist_t.Estimated_Rate_INR_sqft, marker="o", label="Past / demo trend")
    ax.plot(future_t.Year, future_t.Estimated_Rate_INR_sqft, marker="o", linestyle="--", label="Future projection")
    ax.axvline(2026, linestyle=":", alpha=.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated rate (₹ per sq. ft.)")
    ax.set_title(f"{loc_t}, {city_t}: historical and projected trend")
    ax.legend()
    ax.grid(alpha=.25)
    st.pyplot(fig)
    st.caption("2020–2026 values are fabricated demo history; 2027–2030 values are illustrative projections using Linear Regression on the demo trend. They are not observed market data.")

with tab4:
    st.subheader("Model performance on a held-out test split")
    a,b,c = st.columns(3)
    a.metric("Mean Absolute Error (MAE)", inr(mae))
    b.metric("R² score", f"{r2:.3f}")
    c.metric("Test records", f"{len(y_test):,}")
    st.write("**Algorithm:** Linear Regression inside a preprocessing pipeline.")
    st.write("**Categorical encoding:** One-Hot Encoding for city, locality and furnishing.")
    st.write("**Numeric scaling:** StandardScaler for numerical inputs.")
    st.write("**Evaluation:** 80% training data and 20% testing data.")
    st.warning("Because the dataset is synthetic, these metrics only measure performance on generated demo records. They do not prove real-world accuracy.")
    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)
    st.write("**Available cities:** " + ", ".join(sorted(df.City.unique())))
    