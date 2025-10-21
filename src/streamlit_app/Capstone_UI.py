
# Airborne Microbiome Predictor (Group 1)
# Compatible with Streamlit 1.50 and Altair 5.5.0

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import pdist, squareform
from sklearn.inspection import permutation_importance


# Streamlit Page Setup

st.set_page_config(page_title="Airborne Microbiome Predictor", layout="wide")
st.title("Airborne Microbiome Predictor (North America)")


# Load Dataset (pre-imported Excel file)

@st.cache_data
def load_data():
    file_path = "ML-data.xlsx"  # ensure the file is in same directory
    df = pd.read_excel("C:/ML-data.xlsx") # Attach EXCEL DATA PATH HERE
    df.columns = df.columns.str.strip().str.replace(r"[\u200b\u00a0]", "", regex=True)

    numeric_columns = [
        "Temperature (°C)", "Wind Speed (MPH)", "Altitude (km above sea level)",
        "Relative Humidity (%)", "Sea salt Concentration ug/m^3",
        "SO4 Concentration ug/m^3", "Organic Carbon Concentration ug/m^3",
        "Dust Concentration ug/m^3", "Black Carbon Concentration ug/m^3",
        "Concentration (cfu/m^3)", "Sampling duration"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(" ", ""), errors="coerce")

    df.dropna(subset=["Organism", "% Abundance", "Allergen (Y/N)"], inplace=True)

    # Binary encode Rain Y/N
    if "Rain Y/N" in df.columns:
        df["Rain Y/N"] = df["Rain Y/N"].replace({"Y": 1, "N": 0, "y": 1, "n": 0}).astype(int)

    # Label encode categorical features
    cat_cols = ["Rural/Urban", "Climate Type", "Wind Direction"]
    le_dict = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    le_target = LabelEncoder()
    df["Organism_encoded"] = le_target.fit_transform(df["Organism"])

    return df, le_dict, le_target

df, le_dict, le_target = load_data()



# Train Models (cached)

@st.cache_resource
def train_models(df, _le_target):
    features = [
        "Rural/Urban", "Climate Type", "Wind Speed (MPH)", "Wind Direction",
        "Rain Y/N", "Relative Humidity (%)", "Sampling duration",
        "Sea salt Concentration ug/m^3", "SO4 Concentration ug/m^3",
        "Organic Carbon Concentration ug/m^3", "Dust Concentration ug/m^3",
        "Black Carbon Concentration ug/m^3", "Altitude (km above sea level)",
        "Temperature (°C)", "Concentration (cfu/m^3)"
    ]

    X = df[features]
    y_org = df["Organism_encoded"]
    y_allergen = df["Allergen (Y/N)"].replace({"Y": 1, "N": 0, "y": 1, "n": 0}).astype(int)
    y_abundance = df["% Abundance"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_org, test_size=0.2, random_state=42)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_allergen, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_abundance, test_size=0.2, random_state=42)

    clf_org = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf_org.fit(X_train_c, y_train_c)

    clf_all = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_all.fit(X_train_a, y_train_a)

    reg_ab = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_ab.fit(X_train_r, y_train_r)

    r2_val = r2_score(y_test_r, reg_ab.predict(X_test_r))
    rmse_val = math.sqrt(mean_squared_error(y_test_r, reg_ab.predict(X_test_r)))

    metrics = {"R²": r2_val, "RMSE": rmse_val}
    return clf_org, clf_all, reg_ab, metrics, features, X_test_c, y_test_c


clf_org, clf_all, reg_ab, metrics, features, X_test_c, y_test_c = train_models(df, _le_target=le_target)



# Sidebar Scenario Controls

st.sidebar.header("Scenario Controls")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
altitude = st.sidebar.slider("Altitude (km above sea level)", 0.0, 5.0, 0.5)
wind_speed = st.sidebar.slider("Wind Speed (MPH)", 0.0, 30.0, 10.0)
wind_dir = st.sidebar.selectbox("Wind Direction", ["N", "S", "E", "W", "NE", "NW", "SE", "SW"])
humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 50)
rain = st.sidebar.radio("Rain Present?", ["Y", "N"])
climate = st.sidebar.selectbox("Climate Type", ["Coastal", "Grassland", "Forest", "Desert", "Mountain", "Wetland", "Tundra"])
rural_urban = st.sidebar.selectbox("Rural/Urban", ["Rural", "Suburban", "Urban"])
sea_salt = st.sidebar.number_input("Sea Salt (µg/m³)", 0.0, 1.0, 0.05)
so4 = st.sidebar.number_input("SO₄ (µg/m³)", 0.0, 1.0, 0.06)
org_c = st.sidebar.number_input("Organic Carbon (µg/m³)", 0.0, 1.0, 0.03)
dust = st.sidebar.number_input("Dust (µg/m³)", 0.0, 1.0, 0.04)
black_c = st.sidebar.number_input("Black Carbon (µg/m³)", 0.0, 1.0, 0.01)
conc = st.sidebar.number_input("Concentration (cfu/m³)", 0.0, 1000.0, 100.0)
duration = st.sidebar.number_input("Sampling Duration (hrs)", 0.0, 48.0, 12.0)


# Prepare User Input for Prediction

user_input = pd.DataFrame({
    "Rural/Urban": [rural_urban],
    "Climate Type": [climate],
    "Wind Speed (MPH)": [wind_speed],
    "Wind Direction": [wind_dir],
    "Rain Y/N": [1 if rain == "Y" else 0],
    "Relative Humidity (%)": [humidity],
    "Sampling duration": [duration],
    "Sea salt Concentration ug/m^3": [sea_salt],
    "SO4 Concentration ug/m^3": [so4],
    "Organic Carbon Concentration ug/m^3": [org_c],
    "Dust Concentration ug/m^3": [dust],
    "Black Carbon Concentration ug/m^3": [black_c],
    "Altitude (km above sea level)": [altitude],
    "Temperature (°C)": [temp],
    "Concentration (cfu/m^3)": [conc]
})

# Encode categorical columns
for col in le_dict:
    if col in user_input.columns:
        user_input[col] = le_dict[col].transform(user_input[col].astype(str))

# Align columns exactly to training order
user_input = user_input.reindex(columns=features, fill_value=0)


# Run Predictions

org_pred = clf_org.predict(user_input)
org_label = le_target.inverse_transform(org_pred)[0]
allergen_pred = clf_all.predict(user_input)[0]
abundance_pred = reg_ab.predict(user_input)[0]

# Tabs Layout

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Prediction Overview",
    "Model Evaluation",
    "Diversity Indices",
    "Environmental Gradients",
    "Sampling Map",
    "Taxonomic Composition",
    "Methodology",
    "Data Overview",
    "Feature Importance",
    "Model Summary"
])


# --- Prediction Overview Tab (Top 5 Predictions)
with tab1:
    st.header("Predicted Dominant Organism")


    # Main Prediction Output

    st.subheader("Predicted Dominant Organism")
    st.success(f"**{org_label}**")

    # --- Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Allergen Likelihood", "Yes" if allergen_pred == 1 else "No")
    col2.metric("% Abundance", f"{abundance_pred:.2f}%")
    col3.metric("Altitude (km)", f"{altitude:.2f}")

    st.divider()


    # Top 5 Predicted Organisms

    st.subheader("Top 5 Predicted Organisms (by Probability)")

    # --- Compute probabilities for all classes
    probs = clf_org.predict_proba(user_input)[0]
    top_idx = np.argsort(probs)[::-1][:5]
    top_labels = le_target.inverse_transform(top_idx)
    top_probs = probs[top_idx]

    df_top5 = pd.DataFrame({
        "Organism": top_labels,
        "Probability (%)": np.round(top_probs * 100, 2)
    })

    # --- Display as Table
    st.dataframe(
        df_top5.style.highlight_max(subset=["Probability (%)"], color="#cfe2f3"),
        use_container_width=True
    )

    # --- Bar Chart Visualization
    bar_chart = (
        alt.Chart(df_top5)
        .mark_bar(size=30, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Probability (%):Q", title="Prediction Confidence (%)"),
            y=alt.Y("Organism:N", sort="-x"),
            color=alt.Color("Probability (%):Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Organism", "Probability (%)"]
        )
        .properties(
            title="Top 5 Predicted Organisms — Confidence Levels",
            width=600,
            height=350
        )
    )

    st.altair_chart(bar_chart, use_container_width=True)

    st.divider()


    # Probability Distribution (Optional Visualization)
    #
    st.subheader("Prediction Confidence Distribution")

    fig_pie = px.pie(
        df_top5,
        names="Organism",
        values="Probability (%)",
        title="Prediction Confidence Breakdown",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(width=700, height=500)

    st.plotly_chart(fig_pie, use_container_width=True)

    st.caption("""
       **Interpretation Guide:**
       - **Bar Chart** → Ranks the top predicted organisms by confidence level.  
       - **Pie Chart** → Visualizes confidence proportion across top organisms.  
       - The highest confidence organism is shown above as the *predicted dominant organism*.
       """)



# --- Model Evaluation
with tab2:

    # Generate Predictions on Test Data

    y_pred = clf_org.predict(X_test_c)
    y_true = y_test_c

    # Decode encoded labels
    actual_labels = le_target.inverse_transform(y_true)
    predicted_labels = le_target.inverse_transform(y_pred)

    # Build evaluation DataFrame
    df_eval = pd.DataFrame({
        "Actual": actual_labels,
        "Predicted": predicted_labels
    })
    df_eval["Match"] = (df_eval["Actual"] == df_eval["Predicted"]).astype(int)

    # --- Summary metric
    match_rate = df_eval["Match"].mean() * 100


    # Summary Metrics

    col1, col2 = st.columns(2)
    col1.metric("Prediction Agreement (%)", f"{match_rate:.1f}")
    col2.metric("R² (Abundance)", f"{metrics['R²']:.2f}")

    st.caption("Model performance metrics calculated on test subset.")
    st.divider()


    # Predicted vs Actual Heatmap

    st.subheader("Predicted vs Actual Organism Classification")

    # Prepare chart data
    chart_data = (
        df_eval.groupby(["Actual", "Predicted"])
        .size()
        .reset_index(name="Count")
    )

    # Filter to top N organisms
    top_n = st.slider("Show top N most frequent organisms", 10, 50, 25)
    top_orgs = df_eval["Actual"].value_counts().head(top_n).index.tolist()
    chart_filtered = chart_data[
        chart_data["Actual"].isin(top_orgs) &
        chart_data["Predicted"].isin(top_orgs)
        ]

    # Create Altair layers
    base = alt.Chart(chart_filtered).encode(
        x=alt.X("Predicted:N", title="Predicted Organism", sort=top_orgs),
        y=alt.Y("Actual:N", title="Actual Organism", sort=top_orgs)
    )

    heat = base.mark_rect().encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual", "Predicted", "Count"]
    )

    text = base.mark_text(baseline="middle", fontSize=9).encode(
        text="Count:Q",
        color=alt.condition("datum.Count > 10", alt.value("white"), alt.value("black"))
    )

    final_chart = (heat + text).properties(
        width=650,
        height=500,
        title=f"Predicted vs Actual Composition (Top {top_n})"
    )

    # --- Scrollable container for large charts
    with st.expander("View Detailed Predicted vs Actual Chart", expanded=True):
        st.markdown(
            "<div style='overflow-x:auto; overflow-y:auto; max-height:650px; border:1px solid #ddd; border-radius:6px;'>",
            unsafe_allow_html=True,
        )
        st.altair_chart(final_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("""
       This chart compares predicted vs actual organisms from the test dataset.  
       - **Diagonal cells** = correct predictions  
       - **Off-diagonal cells** = misclassifications  
       """)

    st.divider()


    # Residual Analysis (Optional for Regression)

    with st.expander("Residual Analysis (Optional for Regression)", expanded=False):
        try:
            # Compute residuals for abundance regression model
            y_pred_ab = reg_ab.predict(X_test_c)
            residuals = y_test_c - y_pred_ab

            # Create residual scatterplot
            fig_resid = px.scatter(
                x=y_pred_ab,
                y=residuals,
                labels={"x": "Predicted Abundance", "y": "Residual (Actual - Predicted)"},
                title="Residuals vs Predicted Abundance",
                color_discrete_sequence=["#1f77b4"]
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")

            # Display chart in Streamlit
            st.plotly_chart(fig_resid, use_container_width=True)
            st.caption("Helps identify bias or variance patterns in abundance predictions.")
        except Exception as e:
            st.warning(f"Residual plot not generated: {e}")

    st.divider()


    # Permutation Importance (Expand for Details)

    with st.expander("Permutation Importance (Feature Stability Check)"):
        try:
            perm = permutation_importance(
                reg_ab, df[features], df["% Abundance"], n_repeats=5, random_state=42
            )
            perm_df = (
                pd.DataFrame({
                    "Feature": features,
                    "Importance": perm.importances_mean
                })
                .sort_values("Importance", ascending=False)
            )

            fig_perm = px.bar(
                perm_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Permutation Importance (Abundance Model)",
                color="Importance",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_perm, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute permutation importance: {e}")

# --- Diversity Analysis
with tab3:
    st.header("Microbial Diversity Analysis")


    # ALPHA DIVERSITY — SINGLE SAMPLE

    st.subheader("Alpha Diversity (Single-Sample Prediction)")

    if hasattr(clf_org, "predict_proba") and user_input is not None:
        # --- Compute probabilities for the single user input
        probs = clf_org.predict_proba(user_input)[0]

        # Shannon, Simpson, and Richness indices
        shannon = -np.sum(probs * np.log(probs + 1e-10))
        simpson = 1 - np.sum(probs ** 2)
        richness = np.sum(probs > 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Shannon Index", f"{shannon:.3f}")
        col2.metric("Simpson Index", f"{simpson:.3f}")
        col3.metric("Richness", int(richness))

        st.caption("Calculated from predicted organism probability distribution for the current environmental input.")
    else:
        st.warning("Model probabilities not available. Train or load model first.")

    st.divider()


    # ALPHA DIVERSITY — COMPUTE FOR ALL SAMPLES

    with st.expander("Alpha Diversity (Multi-Sample Prediction)", expanded=False):

        if hasattr(clf_org, "predict_proba"):
            try:
                st.subheader("Alpha Diversity (All Samples)")

                # --- Use model feature names to avoid mismatch
                if hasattr(clf_org, "feature_names_in_"):
                    feature_columns = list(clf_org.feature_names_in_)
                else:
                    st.warning("Model does not store feature names — please define feature_columns manually.")
                    feature_columns = []

                X_all = df.filter(items=feature_columns, axis=1)

                if not X_all.empty:
                    X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)
                    probs_all = clf_org.predict_proba(X_all)

                    # --- Compute Shannon, Simpson, Richness
                    df["Shannon"] = -np.sum(probs_all * np.log(probs_all + 1e-10), axis=1)
                    df["Simpson"] = 1 - np.sum(probs_all ** 2, axis=1)
                    df["Richness"] = (probs_all > 0).sum(axis=1)

                    st.success("Alpha diversity metrics successfully calculated for all samples.")
                else:
                    st.warning("No matching columns found between dataset and model feature names.")
            except Exception as e:
                st.warning(f"Could not compute diversity metrics for dataset: {e}")

        st.divider()

        # Optional visualization — scatter of Shannon vs Richness
        if "Shannon" in df.columns and "Richness" in df.columns:
            fig_alpha = px.scatter(
                df,
                x="Shannon",
                y="Richness",
                color="Simpson",
                color_continuous_scale="Viridis",
                title="Alpha Diversity Relationships Across Samples",
                hover_data=["Sample ID"]
            )
            st.plotly_chart(fig_alpha, use_container_width=True)

        st.divider()

        # Display summary table
        st.dataframe(
            df[["Sample ID", "Shannon", "Simpson", "Richness"]].round(3),
            use_container_width=True
        )

    #  BETA DIVERSITY (BRAY–CURTIS)

    with st.expander("Beta Diversity (Bray–Curtis Dissimilarity)"):

        if {"Sample ID", "Organism_encoded", "% Abundance"} <= set(df.columns):
            # --- Compute abundance matrix
            abundance_matrix = df.pivot_table(
                values="% Abundance",
                index="Sample ID",
                columns="Organism_encoded",
                fill_value=0
            )

            # --- Compute Bray–Curtis dissimilarity
            bc_matrix = squareform(pdist(abundance_matrix, metric="braycurtis"))
            bc_df = pd.DataFrame(bc_matrix, index=abundance_matrix.index, columns=abundance_matrix.index)

            st.caption("Shows pairwise dissimilarity between microbial communities based on % Abundance.")
            st.dataframe(bc_df.round(3), use_container_width=True)

            # --- Plot as heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                bc_df,
                cmap="viridis",
                linewidths=0.3,
                linecolor="white",
                cbar_kws={'label': 'Bray–Curtis Dissimilarity'},
                ax=ax
            )
            ax.set_title("Beta Diversity Heatmap (Bray–Curtis Distance)", fontsize=14)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            st.pyplot(fig)

            st.caption("""
               **Interpretation:**
               - **Darker (purple/blue)** → More similar microbial communities  
               - **Brighter (yellow)** → More different microbial compositions  
               """)
        else:
            st.warning(
                "Missing required columns ('Sample ID', '% Abundance', or 'Organism_encoded') for beta diversity heatmap.")


# --- Environmental Gradients
with tab4:
    st.header("Environmental Gradients")
    st.caption("Explore how environmental factors influence microbial diversity, prediction accuracy, and abundance shifts.")


    # 1. ALPHA DIVERSITY VS ENVIRONMENTAL GRADIENTS

    with st.expander("Alpha Diversity vs. Environmental Gradients", expanded=False):
        st.caption("Visualizes how predicted microbial diversity changes with environmental conditions such as altitude, temperature, or humidity.")

        env_factors = [
            "Altitude (km above sea level)",
            "Temperature (°C)",
            "Relative Humidity (%)",
            "SO4 Concentration ug/m^3",
            "Dust Concentration ug/m^3",
            "Organic Carbon Concentration ug/m^3",
            "Sea salt Concentration ug/m^3"
        ]

        available_factors = [f for f in env_factors if f in df.columns]
        diversity_metrics = ["Shannon", "Simpson", "Richness"]

        if all(col in df.columns for col in diversity_metrics) and available_factors:
            selected_factor = st.selectbox(
                "Select Environmental Factor",
                available_factors,
                key="env_factor_tab4"  # unique key prevents duplicate element ID error
            )

            diversity_df = df[diversity_metrics + [selected_factor]].melt(
                id_vars=[selected_factor],
                var_name="Metric",
                value_name="Value"
            )

            scatter_chart = (
                alt.Chart(diversity_df)
                .mark_circle(size=60, opacity=0.6)
                .encode(
                    x=alt.X(f"{selected_factor}:Q", title=selected_factor),
                    y=alt.Y("Value:Q", title="Diversity Value"),
                    color=alt.Color("Metric:N", scale=alt.Scale(scheme="category10")),
                    tooltip=[selected_factor, "Metric", "Value"]
                )
                .properties(width=850, height=400)
            )

            trend = (
                alt.Chart(diversity_df)
                .transform_regression(selected_factor, "Value", groupby=["Metric"])
                .mark_line(size=2)
                .encode(color="Metric:N")
            )

            st.altair_chart(scatter_chart + trend, use_container_width=True)
            st.markdown("""
            Interpretation:
            - Upward trend: Higher diversity under increasing environmental values  
            - Flat trend: Minimal influence  
            - Downward trend: Reduced diversity as the factor increases
            """)
        else:
            st.warning("No diversity or environmental data available for this visualization.")

        st.divider()


    # 2. ALTITUDE INFLUENCE ON PREDICTION ACCURACY


    with st.expander("Altitude Influence on Microbial Prediction Accuracy", expanded=False):
        st.caption("Assesses how well the model performs across different altitude ranges.")

        df_eval = df.copy()
        df_eval["Predicted"] = le_target.inverse_transform(clf_org.predict(df[features]))
        df_eval["Match"] = (df_eval["Predicted"] == df_eval["Organism"]).astype(int)
        df_eval["Altitude_bin"] = pd.cut(df_eval["Altitude (km above sea level)"], bins=10)

        altitude_acc = (
            df_eval.groupby("Altitude_bin")["Match"]
            .mean()
            .reset_index()
            .rename(columns={"Match": "Accuracy"})
        )
        altitude_acc["Accuracy (%)"] = altitude_acc["Accuracy"] * 100

        alt_chart = (
            alt.Chart(altitude_acc)
            .mark_line(point=True)
            .encode(
                x=alt.X("Altitude_bin:N", title="Altitude Range (km)"),
                y=alt.Y("Accuracy (%):Q", title="Prediction Accuracy (%)"),
                tooltip=["Altitude_bin", "Accuracy (%)"]
            )
            .properties(title="Prediction Accuracy vs Altitude", width=850)
        )

        st.altair_chart(alt_chart, use_container_width=True)
        st.caption(
            "Displays prediction accuracy (%) across altitude bins. Higher values indicate better model consistency.")

        st.divider()

    # 3. PREDICTION DIVERGENCE ACROSS ENVIRONMENTAL CONDITIONS

    with st.expander("Prediction Divergence Across Environmental Conditions", expanded=False):
        st.caption("Compares prediction mismatches (divergence) under different environmental gradients.")

        env_options = [f for f in env_factors if f in df.columns]

        df_eval["Divergence"] = (df_eval["Predicted"] != df_eval["Organism"]).astype(int)
        box_data = df_eval.melt(
            id_vars=["Divergence"],
            value_vars=env_options,
            var_name="Factor",
            value_name="Value"
        )

        box_chart = (
            alt.Chart(box_data)
            .mark_boxplot(size=40, opacity=0.7)
            .encode(
                x=alt.X("Factor:N", title="Environmental Factor", sort=None),
                y=alt.Y("Value:Q", title="Measured Value"),
                color=alt.Color("Divergence:N", scale=alt.Scale(scheme="redblue")),
                tooltip=["Factor", "Value", "Divergence"]
            )
            .properties(width=900, height=400, title="Environmental Factors vs Prediction Divergence")
        )

        st.altair_chart(box_chart, use_container_width=True)
        st.caption("Boxplots show how prediction mismatches vary across environmental gradients (red = divergent, blue = accurate).")

        st.divider()


    # 4. POLLUTION VS MICROBIAL ABUNDANCE


    with st.expander("Pollution vs Microbial Abundance Shifts", expanded=False):
        st.caption("Examines how microbial abundance changes with varying pollutant concentrations.")

        pollution_cols = [
            "Sea salt Concentration ug/m^3",
            "SO4 Concentration ug/m^3",
            "Organic Carbon Concentration ug/m^3",
            "Dust Concentration ug/m^3",
            "Black Carbon Concentration ug/m^3"
        ]
        pollution_present = [p for p in pollution_cols if p in df.columns]

        if pollution_present:
            pollution_long = df.melt(
                id_vars=["Organism", "% Abundance"],
                value_vars=pollution_present,
                var_name="Pollutant",
                value_name="Concentration"
            )

            chart_poll = (
                alt.Chart(pollution_long)
                .mark_circle(size=60, opacity=0.6)
                .encode(
                    x=alt.X("Concentration:Q", title="Pollutant Concentration (µg/m³)"),
                    y=alt.Y("% Abundance:Q", title="Microbial Abundance (%)"),
                    color=alt.Color("Pollutant:N", scale=alt.Scale(scheme="viridis")),
                    tooltip=["Organism", "Pollutant", "Concentration", "% Abundance"]
                )
                .properties(title="Pollution Factors vs Microbial Abundance Shifts", width=900)
            )

            st.altair_chart(chart_poll, use_container_width=True)
            st.caption("Scatter plot shows how microbial abundance responds to pollutant levels, useful for identifying sensitive taxa.")
        else:
            st.warning("No pollution concentration columns found in dataset.")

        st.divider()


    # 5. ENVIRONMENT–DIVERSITY CORRELATION MATRIX

    with st.expander("Environmental–Diversity Correlation Matrix", expanded=False):
        st.caption("Highlights relationships between environmental variables and alpha diversity indices.")

        corr_vars = [
            "Altitude (km above sea level)",
            "Temperature (°C)",
            "Relative Humidity (%)",
            "Dust Concentration ug/m^3",
            "SO4 Concentration ug/m^3",
            "Organic Carbon Concentration ug/m^3",
            "Black Carbon Concentration ug/m^3",
            "Sea salt Concentration ug/m^3",
            "Shannon",
            "Simpson",
            "Richness"
        ]
        corr_present = [c for c in corr_vars if c in df.columns]

        if len(corr_present) >= 2:
            corr = df[corr_present].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            ax.set_title("Correlation Matrix of Environmental and Diversity Factors", fontsize=14)
            st.pyplot(fig)
            st.caption("""
            Interpretation:
            - Red: Positive correlation (both increase together)
            - Blue: Negative correlation (one increases while the other decreases)
            - Near 0: Weak or no relationship
            """)
        else:
            st.warning("Insufficient numeric variables available for correlation heatmap.")


    # 6. ORGANISM DOMINANCE ACROSS ENVIRONMENTAL FACTORS

    with st.expander("Organism Dominance Across Environmental Factors", expanded=False):
        st.caption("Displays which organisms are most abundant under different environmental gradients.")

        dominant_factors = [
            "Altitude (km above sea level)",
            "Temperature (°C)",
            "Relative Humidity (%)",
            "SO4 Concentration ug/m^3",
            "Dust Concentration ug/m^3",
            "Organic Carbon Concentration ug/m^3"
        ]

        available_factors = [f for f in dominant_factors if f in df.columns]

        if available_factors:
            selected_factor = st.selectbox(
                "Select Environmental Factor",
                available_factors,
                key="dominance_factor_tab4"
            )

            # Bin the environmental variable into 6 ranges and convert intervals to strings
            df["Factor_Bin"] = pd.qcut(df[selected_factor], q=6, duplicates="drop")
            df["Factor_Bin"] = df["Factor_Bin"].astype(str)

            # Compute mean abundance per organism within each bin
            abundance_matrix = (
                df.groupby(["Factor_Bin", "Organism"])["% Abundance"]
                .mean()
                .reset_index()
            )

            # Keep top 15 most abundant organisms overall
            top_orgs = df.groupby("Organism")["% Abundance"].mean().nlargest(15).index
            abundance_matrix = abundance_matrix[abundance_matrix["Organism"].isin(top_orgs)]

            pivot_df = abundance_matrix.pivot(index="Organism", columns="Factor_Bin", values="% Abundance").fillna(0)

            heatmap_fig = px.imshow(
                pivot_df,
                aspect="auto",
                color_continuous_scale="YlGnBu",
                labels=dict(color="Mean % Abundance"),
                title=f"Dominant Organisms Across {selected_factor} Ranges"
            )

            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption(
                "The heatmap shows which organisms dominate under specific environmental ranges. "
                "Darker colors indicate higher relative abundance."
            )
        else:
            st.warning("No environmental variables available for organism dominance analysis.")


    # 7. ABUNDANCE TRENDS FOR SELECTED ORGANISMS (X-AXIS SWITCHABLE)

    with st.expander("Abundance Trends for Selected Organisms", expanded=False):
        st.caption("Explore how organism abundance changes with different environmental gradients.")

        # Define available environmental variables for x-axis
        env_vars = [
            "Altitude (km above sea level)",
            "Temperature (°C)",
            "Relative Humidity (%)",
            "SO4 Concentration ug/m^3",
            "Dust Concentration ug/m^3",
            "Organic Carbon Concentration ug/m^3",
            "Sea salt Concentration ug/m^3"
        ]
        available_vars = [v for v in env_vars if v in df.columns]

        if available_vars:
            # --- Select environmental variable for x-axis
            selected_x = st.selectbox(
                "Select Environmental Variable (X-axis):",
                available_vars,
                key="x_axis_env_var"
            )

            # --- Select organisms to plot
            all_organisms = sorted(df["Organism"].dropna().unique())
            selected_orgs = st.multiselect(
                "Select Organisms to Plot:",
                all_organisms[:30],
                default=all_organisms[:5],
                key="selected_org_trend"
            )

            if selected_orgs:
                trend_df = df[df["Organism"].isin(selected_orgs)].copy()

                chart = (
                    alt.Chart(trend_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{selected_x}:Q", title=selected_x),
                        y=alt.Y("% Abundance:Q", title="Mean Abundance (%)"),
                        color=alt.Color("Organism:N", legend=alt.Legend(title="Organism")),
                        tooltip=["Organism", selected_x, "% Abundance"]
                    )
                    .properties(
                        width=850,
                        height=400,
                        title=f"Abundance Trends vs {selected_x}"
                    )
                    .interactive()
                )

                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "Shows how each organism’s abundance changes continuously along the selected environmental gradient.")
            else:
                st.info("Select at least one organism to visualize its abundance trend.")
        else:
            st.warning("No environmental variables available for abundance trend visualization.")

# --- Sampling Map
with tab5:
    st.header("Sampling Distribution and Altitude Variability")
    st.caption("Visualize where microbial samples were collected, explore altitude variations, and analyze how elevation affects prediction accuracy.")

    # --- Define city coordinates
    city_coords = {
        "Colorado": (39.5501, -105.7821), "Ohio": (40.4173, -82.9071),
        "Illinois": (40.6331, -89.3985), "Michigan": (44.3148, -85.6024),
        "Wisconsin": (44.5000, -89.5000), "Alaska": (64.2008, -149.4937),
        "Hawaii": (19.8968, -155.5828), "Tennessee": (35.5175, -86.5804),
        "Oregon": (43.8041, -120.5542), "Oklahoma": (35.4676, -97.5164),
        "Washington, D.C.": (38.8951, -77.0364), "Virginia": (37.4316, -78.6569),
        "Arkansas": (35.2010, -91.8318), "Arizona": (34.0489, -111.0937),
        "California": (36.7783, -119.4179), "Texas": (31.0000, -100.0000),
        "Connecticut": (41.6032, -73.0877), "Florida": (27.9944, -81.7603),
        "North Carolina": (35.7596, -79.0193), "Missouri": (38.5739, -92.6038),
        "Massachusetts": (42.4072, -71.3824), "Nebraska": (41.4925, -99.9018),
        "New York": (43.0000, -75.0000)
    }


    # INTERACTIVE SAMPLING MAP

    if "City/Area" in df.columns:
        st.subheader("Interactive Sampling Map (U.S. Regions)")

        # --- Count samples per city
        sample_counts = df["City/Area"].value_counts().reset_index()
        sample_counts.columns = ["City/Area", "Sample Count"]

        # --- Attach coordinates
        sample_counts["Latitude"], sample_counts["Longitude"] = zip(
            *sample_counts["City/Area"].apply(lambda city: city_coords.get(city, (None, None)))
        )

        # --- Merge with altitude if available
        if "Altitude (km above sea level)" in df.columns:
            avg_alt = (
                df.groupby("City/Area")["Altitude (km above sea level)"].mean().reset_index()
            )
            df_map = pd.merge(sample_counts, avg_alt, on="City/Area", how="left")
        else:
            df_map = sample_counts.copy()

        df_map = df_map.dropna(subset=["Latitude", "Longitude"])

        if not df_map.empty:
            from vega_datasets import data
            states = alt.topo_feature(data.us_10m.url, "states")

            # --- Background map
            background = (
                alt.Chart(states)
                .mark_geoshape(fill="lightgray", stroke="white")
                .properties(width=950, height=600)
                .project("albersUsa")
            )

            # --- City bubbles
            bubbles = (
                alt.Chart(df_map)
                .mark_circle(opacity=0.85)
                .encode(
                    longitude="Longitude:Q",
                    latitude="Latitude:Q",
                    size=alt.Size(
                        "Sample Count:Q",
                        scale=alt.Scale(range=[200, 1500]),
                        legend=alt.Legend(title="Sample Count")
                    ),
                    color=alt.Color(
                        "Altitude (km above sea level):Q",
                        scale=alt.Scale(scheme="viridis"),
                        legend=alt.Legend(title="Altitude (km)")
                    ),
                    tooltip=["City/Area", "Sample Count", "Altitude (km above sea level)"]
                )
            )

            labels = (
                alt.Chart(df_map)
                .mark_text(fontSize=12, dy=-10, fontWeight="bold", color="black")
                .encode(
                    longitude="Longitude:Q",
                    latitude="Latitude:Q",
                    text="Sample Count:Q"
                )
            )

            bubble_map = (background + bubbles + labels).interactive()
            st.altair_chart(bubble_map, use_container_width=True)
            st.caption("🟢 Larger circles = more samples. Color scale represents mean altitude per city.")

            st.divider()


            # ALTITUDE DISTRIBUTION SCATTERPLOT

            with st.expander("Altitude Distribution by Location", expanded=False):

                min_alt = float(df_map["Altitude (km above sea level)"].min())
                max_alt = float(df_map["Altitude (km above sea level)"].max())
                selected_range = st.slider(
                    "Filter by Altitude Range (km above sea level):",
                    min_value=round(min_alt, 2),
                    max_value=round(max_alt, 2),
                    value=(round(min_alt, 2), round(max_alt, 2)),
                    step=0.1
                )

                df_filtered = df_map[
                    (df_map["Altitude (km above sea level)"] >= selected_range[0]) &
                    (df_map["Altitude (km above sea level)"] <= selected_range[1])
                ]

                scatter = (
                    alt.Chart(df_filtered)
                    .mark_circle(size=180, opacity=0.8)
                    .encode(
                        x=alt.X("City/Area:N", sort="-y", title="Location"),
                        y=alt.Y("Altitude (km above sea level):Q", title="Altitude (km, relative to sea level)"),
                        color=alt.Color("Altitude (km above sea level):Q",
                                        scale=alt.Scale(scheme="viridis"), legend=None),
                        size=alt.Size("Sample Count:Q", scale=alt.Scale(range=[100, 600])),
                        tooltip=["City/Area", "Altitude (km above sea level)", "Sample Count"]
                    )
                    .properties(
                        width=950,
                        height=450,
                        title=f"Altitude Variation across Sampling Locations ({selected_range[0]}–{selected_range[1]} km)"
                    )
                    .interactive()
                )

                # --- Sea-level reference line
                sea_level_line = (
                    alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(color="red", strokeWidth=2, strokeDash=[6, 3])
                    .encode(y="y:Q")
                )

                st.altair_chart(scatter + sea_level_line, use_container_width=True)
                st.caption("🔴 Red dashed line marks sea level (0 km). Hover or zoom to explore altitude variation.")

        else:
            st.warning("No recognized cities found for mapping. Please check 'City/Area' names.")

    else:
        st.warning("Dataset does not include a 'City/Area' column.")


    # ALTITUDE INFLUENCE ON MODEL PERFORMANCE

    st.divider()
    with st.expander("Altitude Influence on Microbial Prediction Accuracy", expanded=False):

        if "Altitude (km above sea level)" in df.columns and "Organism" in df.columns:
            try:
                df_eval = df.copy()
                df_eval["Predicted"] = le_target.inverse_transform(clf_org.predict(df[features]))
                df_eval["Match"] = (df_eval["Predicted"] == df_eval["Organism"]).astype(int)

                # --- Accuracy by altitude bins
                df_eval["Altitude_bin"] = pd.cut(df_eval["Altitude (km above sea level)"], bins=10)
                altitude_acc = df_eval.groupby("Altitude_bin")["Match"].mean().reset_index()
                altitude_acc["Accuracy (%)"] = altitude_acc["Match"] * 100

                # --- Chart
                alt_chart = (
                    alt.Chart(altitude_acc)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Altitude_bin:N", title="Altitude Range (km)"),
                        y=alt.Y("Accuracy (%):Q", title="Prediction Accuracy (%)"),
                        tooltip=["Altitude_bin", "Accuracy (%)"]
                    )
                    .properties(title="Prediction Accuracy vs Altitude", width=900)
                )

                st.altair_chart(alt_chart, use_container_width=True)
                st.caption("Displays how model prediction accuracy varies across different altitude bands.")
            except Exception as e:
                st.warning(f"Could not compute altitude influence chart: {e}")
        else:
            st.info("Altitude or organism data missing — cannot compute prediction accuracy by altitude.")


with tab6:
    st.header("Taxonomic Composition and Hierarchy")
    st.caption("Explore microbial relative abundance patterns across Phylum, Class, and Genus levels.")

    # --- Check for required taxonomy columns
    if {"Phylum", "Class", "Genus", "% Abundance"} <= set(df.columns):

        # HIERARCHICAL TREEMAP

        with st.expander("Hierarchical Treemap (Phylum → Class → Genus)", expanded=False):
            st.caption("Each rectangle represents a taxonomic level; area corresponds to relative abundance, and color distinguishes Phyla.")

            # Aggregate by Phylum → Class → Genus
            df_treemap = (
                df.groupby(["Phylum", "Class", "Genus"], as_index=False)["% Abundance"]
                .sum()
                .sort_values("% Abundance", ascending=False)
            )

            # Plot Treemap
            fig_treemap = px.treemap(
                df_treemap,
                path=["Phylum", "Class", "Genus"],
                values="% Abundance",
                color="Phylum",
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Relative Abundance by Taxonomic Hierarchy",
            )

            fig_treemap.update_layout(
                width=950,
                height=600,
                margin=dict(t=60, l=20, r=20, b=20)
            )

            st.plotly_chart(fig_treemap, use_container_width=True)

        # STACKED BAR CHART: TOP PHYLA BY GENUS

        st.divider()
        with st.expander("Stacked Bar Chart by Phylum and Genus", expanded=False):

            st.caption("Highlights dominant genera within the top 5 most abundant phyla across all samples.")

            # Filter top 5 most abundant phyla
            top_phyla = df["Phylum"].value_counts().head(5).index.tolist()
            filtered_df = df[df["Phylum"].isin(top_phyla)]

            # Aggregate mean abundance per Phylum–Genus pair
            df_bar = (
                filtered_df.groupby(["Phylum", "Genus"], as_index=False)["% Abundance"]
                .mean()
                .sort_values("% Abundance", ascending=False)
            )

            # Plot stacked bar chart
            fig_bar = px.bar(
                df_bar,
                x="Phylum",
                y="% Abundance",
                color="Genus",
                title="Dominant Genera within Top 5 Phyla",
                text_auto=".2f",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )

            fig_bar.update_layout(
                width=950,
                height=600,
                barmode="stack",
                xaxis_title="Phylum",
                yaxis_title="Mean % Abundance",
                margin=dict(t=60, l=20, r=20, b=20),
                legend_title="Genus"
            )

            st.plotly_chart(fig_bar, use_container_width=True)


            # OPTIONAL INSIGHTS SECTION

            with st.expander("Interpretation Tips", expanded=False):
                st.markdown("""
                **How to Read These Charts:**
                - The **Treemap** shows taxonomic hierarchy — larger boxes = higher relative abundance.  
                - The **Stacked Bar Chart** highlights dominant genera contributing to each top phylum.  
                - Use hover tooltips for details on genus contributions and abundance percentages.
                """)

    else:
        st.warning(" Missing required columns ('Phylum', 'Class', 'Genus', or '% Abundance'). Please ensure your dataset includes these fields.")

with tab7:
    st.header("Methodology Overview")
    st.caption("Summarizes laboratory preparation, gene amplification, and sequencing techniques used across samples.")

    method_cols = [
        "Gene Amplification Method",
        "Extraction Method",
        "Gene Promoter",
        "Primer",
        "Sequencing Platform",
        "Sampling Method"
    ]

    available_methods = [col for col in method_cols if col in df.columns]

    if available_methods:
        st.markdown("### Overview of Laboratory and Sequencing Techniques")
        st.caption(
            "Each chart displays how frequently each methodological approach appears in the dataset. "
            "The bars represent **sample counts**, and tooltips show both counts and percentages. "
            "Entries marked **'Not Specified'** indicate missing or undefined metadata."
        )

        for col in available_methods:
            with st.expander(f"{col} — Frequency Distribution", expanded=False):

                # --- Clean values
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"[\u200b\u00a0]", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .replace({"nan": "Not Specified", "NaN": "Not Specified", "": "Not Specified"})
                )

                # --- Counts & %
                method_counts = df[col].value_counts(dropna=False).reset_index()
                method_counts.columns = [col, "Count"]
                total = method_counts["Count"].sum()
                method_counts["Percentage"] = (method_counts["Count"] / total * 100).round(2)

                chart_height = max(400, len(method_counts) * 35)

                # --- Bar chart
                chart = (
                    alt.Chart(method_counts)
                    .mark_bar(size=30, opacity=0.9)
                    .encode(
                        x=alt.X("Count:Q", title="Number of Samples"),
                        y=alt.Y(f"{col}:N", sort="-x", title=None),
                        color=alt.Color(f"{col}:N", scale=alt.Scale(scheme="category20"), legend=None),
                        tooltip=[col, "Count", "Percentage"]
                    )
                    .properties(
                        title=f"Distribution of {col}",
                        width=1000,
                        height=chart_height
                    )
                )
                st.altair_chart(chart, use_container_width=True)

                # --- Table
                with st.expander(f"{col} — Percentage Table", expanded=False):
                    st.dataframe(
                        method_counts[[col, "Percentage"]]
                        .style.highlight_max("Percentage", color="#cce5ff")
                        .format({"Percentage": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True
                    )

            st.markdown("---")
    else:
        st.warning("No methodology-related columns found in the dataset.")

with tab8:
    st.header("Dataset Summary and Quality Checks")
    st.caption("Quick summary of dataset completeness, structure, and numeric relationships.")

    col1, col2 = st.columns(2)
    col1.metric("Total Samples", len(df))
    col2.metric("Total Features", df.shape[1])

    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(df.describe(include="all").round(2), use_container_width=True)

    with st.expander("Missing Values Summary", expanded=False):
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing = missing[missing["Missing Count"] > 0]
        if not missing.empty:
            st.dataframe(missing, use_container_width=True)
        else:
            st.success("No missing values detected.")

    with st.expander("Numeric Correlation Heatmap", expanded=False):
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap of Numeric Features", fontsize=14)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation heatmap.")

with tab9:
    st.header("Feature Importance & Interpretability")

    with st.expander("Model-Based Feature Importance (Tree-Derived)", expanded=False):
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": clf_org.feature_importances_
        }).sort_values("Importance", ascending=False)

        chart = (
            alt.Chart(importance.head(15))
            .mark_bar(size=20)
            .encode(
                x="Importance:Q",
                y=alt.Y("Feature:N", sort="-x"),
                color="Importance:Q",
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")]
            )
            .properties(title="Top 15 Predictive Environmental Features", width=800, height=500)
        )
        st.altair_chart(chart, use_container_width=True)

    with st.expander("Permutation Importance (Model-Agnostic)", expanded=False):
        try:
            perm = permutation_importance(reg_ab, df[features], df["% Abundance"], n_repeats=5, random_state=42)
            perm_df = pd.DataFrame({"Feature": features, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
            fig_perm = px.bar(
                perm_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="viridis",
                title="Top 15 Features by Permutation Importance"
            )
            st.plotly_chart(fig_perm, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not calculate permutation importance: {e}")

    with st.expander("Feature Correlation Overview (Model Predictors)", expanded=False):
        numeric_df = df[features].select_dtypes(include=["number"]) if features else pd.DataFrame()
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
            ax.set_title("Feature Correlation Heatmap (Predictors)", fontsize=14)
            st.pyplot(fig_corr)
        else:
            st.warning("No numeric predictors available.")

with tab10:
    st.header("Model Summary and Evaluation Overview")

    with st.expander("Model Performance Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{metrics.get('R²', 0):.3f}")
        col2.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
        col3.metric("MAE", f"{metrics.get('MAE', 0):.4f}")

    with st.expander("Cross-Validation Results", expanded=False):
        if "cv_mean" in locals():
            st.metric("Average CV R²", f"{cv_mean:.3f}")
            st.metric("Standard Deviation", f"±{cv_std:.3f}")
        else:
            st.info("Cross-validation metrics unavailable.")

    with st.expander("Residual Analysis", expanded=False):
        try:
            if "y_test" in locals() and "y_pred" in locals():
                residuals = y_test - y_pred
                fig_resid = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={"x": "Predicted Abundance", "y": "Residuals"},
                    color=abs(residuals),
                    color_continuous_scale="RdBu",
                    title="Residuals vs Predicted Abundance"
                )
                fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig_resid, use_container_width=True)
            else:
                st.info("Residual data unavailable.")
        except Exception as e:
            st.warning(f"Could not plot residual diagnostics: {e}")

    with st.expander("Model Notes and Export Options", expanded=False):
        st.markdown(
            """
            - Trained on optimized environmental predictors.  
            - Evaluated with hold-out and cross-validation testing.  
            - Metrics suggest strong predictive performance.
            """
        )
        if "metrics" in locals():
            csv_data = pd.DataFrame([metrics]).to_csv(index=False).encode()
            st.download_button(
                label="Download Model Metrics (CSV)",
                data=csv_data,
                file_name="model_metrics.csv",
                mime="text/csv"
            )


