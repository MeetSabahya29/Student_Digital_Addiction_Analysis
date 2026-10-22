import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Student Digital Addiction Analysis", page_icon="📱", layout="wide")
st.title("📱 Student Digital Addiction Analysis")
st.caption("An educational data analytics project for exploring student digital habits. It is not a medical or clinical diagnostic tool.")

DEFAULT_DATA = Path(__file__).parent / "data" / "student_digital_habits_synthetic.csv"
REQUIRED = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "sleep_hours",
            "study_hours", "notifications_per_day", "academic_performance_score", "risk_level"]

@st.cache_data
def load_default(path):
    return pd.read_csv(path)

uploaded = st.sidebar.file_uploader("Optional: upload your own CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.info("Using your uploaded dataset.")
else:
    df = load_default(DEFAULT_DATA)
    st.sidebar.info("Using included synthetic demo dataset (1,200 rows).")

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Your CSV is missing required columns: " + ", ".join(missing))
    st.markdown("Required columns: `" + "`, `".join(REQUIRED) + "`")
    st.stop()

df = df.copy()
numeric_cols = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "sleep_hours",
                "study_hours", "notifications_per_day", "academic_performance_score"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["risk_level"] = df["risk_level"].astype(str).str.title()
df = df.dropna(subset=["risk_level"])
if df.empty:
    st.error("No usable rows found in the uploaded dataset.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Explore Data", "Relationships", "ML Model", "Student Risk Tool"])

with tab1:
    st.subheader("Dataset overview")
    a,b,c,d = st.columns(4)
    a.metric("Students / records", f"{len(df):,}")
    b.metric("Average screen time", f"{df['daily_screen_time_hours'].mean():.1f} hrs/day")
    c.metric("Average sleep", f"{df['sleep_hours'].mean():.1f} hrs/night")
    d.metric("High-risk label share", f"{(df['risk_level']=='High').mean()*100:.1f}%")
    st.markdown("### Risk category distribution")
    counts = df["risk_level"].value_counts().reindex(["Low","Moderate","High"]).fillna(0)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Dataset risk category")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    st.markdown("### Sample records")
    st.dataframe(df.head(15), use_container_width=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "student_digital_habits.csv", "text/csv")

with tab2:
    st.subheader("Explore patterns")
    metric = st.selectbox("Choose a measure", numeric_cols, index=0)
    fig, ax = plt.subplots()
    df[metric].dropna().plot(kind="hist", bins=20, ax=ax)
    ax.set_xlabel(metric.replace("_"," ").title())
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    if "gender" in df.columns:
        st.markdown("### Average usage by gender (descriptive only)")
        grouped = df.groupby("gender", dropna=False)[["daily_screen_time_hours","social_media_hours","sleep_hours","study_hours"]].mean().round(2)
        st.dataframe(grouped, use_container_width=True)
    st.markdown("### Filter records")
    risk_filter = st.multiselect("Risk category", sorted(df["risk_level"].dropna().unique()), default=sorted(df["risk_level"].dropna().unique()))
    filtered = df[df["risk_level"].isin(risk_filter)]
    st.write(f"{len(filtered):,} records match the filter.")
    st.dataframe(filtered, use_container_width=True)

with tab3:
    st.subheader("Relationships between habits")
    xcol = st.selectbox("X-axis", numeric_cols, index=0, key="x")
    ycol = st.selectbox("Y-axis", [x for x in numeric_cols if x != xcol], index=3, key="y")
    fig, ax = plt.subplots()
    for label, subset in df.groupby("risk_level"):
        ax.scatter(subset[xcol], subset[ycol], label=label, alpha=0.55)
    ax.set_xlabel(xcol.replace("_"," ").title())
    ax.set_ylabel(ycol.replace("_"," ").title())
    ax.legend()
    st.pyplot(fig)
    st.caption("Correlation and association do not prove that digital-device use causes a change in sleep or academic performance.")
    st.markdown("### Correlation matrix")
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=[x.replace("_"," ") for x in corr.columns], rotation=75, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=[x.replace("_"," ") for x in corr.index])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

with tab4:
    st.subheader("Train a baseline Machine Learning model")
    st.warning("The included demo labels are generated using a simple educational rule, so model scores are for pipeline demonstration only—not evidence of real-world diagnostic accuracy.")
    target = "risk_level"
    feature_candidates = [c for c in numeric_cols if c in df.columns]
    if "age" in df.columns: feature_candidates.append("age")
    if "gender" in df.columns: feature_candidates.append("gender")
    feature_cols = st.multiselect("Features", feature_candidates, default=feature_candidates)
    if len(feature_cols) < 1:
        st.info("Select at least one feature to train the model.")
    elif df[target].nunique() < 2:
        st.error("At least two risk categories are required to train the model.")
    else:
        model_df = df[feature_cols + [target]].dropna()
        X, y = model_df[feature_cols], model_df[target]
        if y.value_counts().min() < 2:
            st.error("Each risk category needs at least two records for a stratified split.")
        else:
            cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
            num_cols = [c for c in X.columns if c not in cat_cols]
            transformers = []
            if num_cols:
                transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
            if cat_cols:
                transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
            preprocessor = ColumnTransformer(transformers)
            clf = Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))])
            test_size = st.slider("Test set size", min_value=0.2, max_value=0.4, value=0.25, step=0.05)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            st.metric("Test accuracy", f"{accuracy_score(y_test, pred)*100:.1f}%")
            st.text("Classification report\n" + classification_report(y_test, pred, zero_division=0))
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
            ax.set_title("Confusion matrix")
            st.pyplot(fig)
            st.caption("For a real study, use a validated questionnaire/label definition, representative consented data, and a separate external test set.")

with tab5:
    st.subheader("Explore a hypothetical student's habits")
    st.write("Adjust the values to see the model's estimated category. This is an educational demonstration, not a diagnosis.")
    features = []
    for col in ["daily_screen_time_hours","social_media_hours","gaming_hours","sleep_hours","study_hours","notifications_per_day","academic_performance_score"]:
        if col in df.columns:
            defaults = {
                "daily_screen_time_hours": (0.5,14.0,6.0,0.5),
                "social_media_hours": (0.0,8.0,2.5,0.5),
                "gaming_hours": (0.0,8.0,1.0,0.5),
                "sleep_hours": (3.0,10.0,7.0,0.5),
                "study_hours": (0.0,8.0,3.0,0.5),
                "notifications_per_day": (0,300,80,5),
                "academic_performance_score": (0,100,75,5)
            }[col]
            val = st.slider(col.replace("_"," ").title(), defaults[0], defaults[1], defaults[2], defaults[3], key="risk_"+col)
            features.append((col,val))
    if "age" in df.columns:
        features.append(("age", st.slider("Age", 17, 30, 21)))
    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().astype(str).unique().tolist())
        features.append(("gender", st.selectbox("Gender (optional)", genders)))
    if st.button("Estimate category", type="primary"):
        # A model is trained from the available data with the same selected feature set.
        cols = [x[0] for x in features if x[0] in feature_candidates]
        if cols and df["risk_level"].nunique() >= 2:
            train_df = df[cols + ["risk_level"]].dropna()
            cat = train_df[cols].select_dtypes(include=["object","category"]).columns.tolist()
            num = [x for x in cols if x not in cat]
            trs = []
            if num: trs.append(("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num))
            if cat: trs.append(("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat))
            pipe = Pipeline([("preprocess",ColumnTransformer(trs)),("model",RandomForestClassifier(n_estimators=150,random_state=42,class_weight="balanced"))])
            pipe.fit(train_df[cols], train_df["risk_level"])
            sample = pd.DataFrame([{k:v for k,v in features if k in cols}])
            prediction = pipe.predict(sample)[0]
            st.success(f"Estimated dataset category: {prediction}")
            st.caption("This result reflects the labels and limitations of the dataset provided; it is not a clinical assessment.")
        else:
            st.error("The dataset needs usable features and at least two risk categories.")

st.divider()
st.caption("Privacy note: Do not upload identifiable student information without permission. Use anonymized records and follow your institution's ethics and data-protection requirements.")
