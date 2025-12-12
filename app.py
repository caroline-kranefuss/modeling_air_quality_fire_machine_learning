import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Tuple, Any, Dict, List

# --- SKLEARN / STATSMODEL IMPORTS ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, log_loss, 
    confusion_matrix, roc_curve, roc_auc_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure Matplotlib figures are closed to prevent memory issues
plt.close('all') 

# ==============================================================================
# 0. DATA LOADING
# ==============================================================================

@st.cache_data
def load_data(index_col: int | None = None) -> pd.DataFrame | None:
    """Loads the dataset and caches it. index_col=0 for PCA, None for others."""
    data_path = 'air_quality_weather_fires.csv'
    if not os.path.exists(data_path):
        st.error(f"Error: Data file not found at '{data_path}'. Please ensure the file is in the current directory.")
        return None
    try:
        df = pd.read_csv(data_path, index_col=index_col)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ==============================================================================
# 1. K-MEANS CLUSTERING (Unsupervised Analysis)
# ==============================================================================

def run_kmeans_analysis(data: pd.DataFrame) -> Tuple[px.line, px.line, pd.DataFrame, pd.Series]:
    """
    Runs K-Means clustering analysis, including K selection methods, 
    and returns plots and cluster center characteristics.
    """
    df = data.copy()
    
    # Recoding weather codes (as in original script)
    df['weather_code'] = df['weather_code'].replace(['1', 'Clear sky', 'Mainly clear'], 'clear')
    df = df[df.weather_code != '2']
    df['weather_code'] = df['weather_code'].replace(
        ['3', 'Overcast', 'Partly cloudy'], 'cloudy')
    df['weather_code'] = df['weather_code'].replace(
        ['51', '53', '55', '61', '63', '65', 
         'Dense drizzle', 'Heavy rain', 'Light drizzle', 'Moderate drizzle', 'Moderate rain', 'Slight rain'], 'rainy')
    df['weather_code'] = df['weather_code'].replace(
        ['71', '73', '75', 'Heavy snow fall', 'Moderate snow fall', 'Slight snow fall'], 'snowy')

    # Corrected X_cols based on previous interaction (assuming original data had full names)
    X_cols = ['latitude', 'longitude', 'temperature_2m_mean', 
              'wind_speed_10m_mean', 'precipitation_sum', 
              'fires_within_50km', 'fires_within_100km', 
              'distance_to_fire_km', 'fire_brightness']
    target = 'PM25'

    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=X_cols + [target], inplace=True)

    X_data = df[X_cols]
    # y_data = df[target].values # Not needed for pure clustering

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(random_state=42, n_init='auto'))
    ])
    
    # 1. Elbow Method (WCSS)
    K_values = list(range(1, 11))
    wcss = []
    for k in K_values:
        pipe.set_params(kmeans__n_clusters=k)
        pipe.fit(X_data)
        wcss.append(pipe['kmeans'].inertia_)

    fig_elbow = px.line(x=K_values, y=wcss, markers=True, title="1a. Elbow Method: WCSS vs. K", 
                        labels={"x": "Number of Clusters (K)", "y": "WCSS (Inertia)"}, height=350)

    # 2. Silhouette Scores
    sil_scores = []
    for k in range(2, 11):
        pipe.set_params(kmeans__n_clusters=k)
        pipe.fit(X_data)
        labels = pipe['kmeans'].labels_
        sil_scores.append(silhouette_score(X_data, labels))

    fig_sil = px.line(x=list(range(2, 11)), y=sil_scores, markers=True, title="1b. Silhouette Scores for different K", 
                      labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"}, height=350)
    
    # Final K-Means Model (K=2 chosen from analysis)
    final_k = 2
    pipe.set_params(kmeans__n_clusters=final_k)
    pipe.fit(X_data)
    df['cluster'] = pipe['kmeans'].labels_
    
    # 3. Scaled Cluster Centers
    cluster_centers_df = pd.DataFrame(pipe['kmeans'].cluster_centers_, columns=X_cols)
    cluster_centers_df.index.name = "Cluster"

    # 4. Average PM25 per Cluster
    pm25_means = df.groupby('cluster')[target].mean()
    pm25_means.name = "PM25_Mean"

    return (fig_elbow, fig_sil, cluster_centers_df, pm25_means)


# ==============================================================================
# 2. KNN REGRESSION (From knn_streamlit.py)
# ==============================================================================

def run_knn_regression(data: pd.DataFrame) -> Tuple[pd.DataFrame, px.line, plt.Figure, plt.Figure, int, float]:
    # Analysis logic copied directly from knn_streamlit.py
    df = data.copy()
    
    def calculate_metrics(y_true, y_pred, best_k):
        rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_score_final = r2_score(y_true, y_pred)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Best K', 'RMSE', 'R² Score'],
            'Value': [best_k, f"{rmse_score:.3f} (AQI units)", f"{r2_score_final:.3f}"],
            'Interpretation': ['Optimal number of neighbors used in the final model.',
                               'Average prediction error in AQI units.',
                               'Percentage of AQI variance explained by the model.']
        }).set_index('Metric')
        return metrics_df, rmse_score

    # Feature Engineering
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m_dominant'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m_dominant'] / 360)

    target = 'AQI_PM25'
    # Dropping columns based on user VIF/Multicollinearity assessment in interpretation notes
    predictors = [
        'latitude', 'longitude', 'temperature_2m_mean', 'relative_humidity_2m_mean',
        'wind_speed_10m_mean', 'wind_dir_sin', 'wind_dir_cos', 'precipitation_hours',
        'distance_to_fire_km', 'fire_brightness', 
        'fires_within_50km', 'fires_within_100km', 'month_sin', 'month_cos', 'wildfire_season'
        # Removed 'fire_frp' (as per user note, highly correlated with 'fire_brightness')
    ]

    df = df.dropna(subset=predictors + [target])
    X = df[predictors]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Correlation Heatmap
    corr = X.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("2d. Predictor Correlation Heatmap", fontsize=14)
    plt.close(fig_corr)

    # Hyperparameter Tuning (GridSearch)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(weights="distance"))])
    param_grid = {'knn__n_neighbors': list(range(1, 41, 2))}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_k = grid.best_params_['knn__n_neighbors']
    
    # 2. CV Plot (Negative MSE vs. K)
    results_df = pd.DataFrame(grid.cv_results_)
    results_df["k"] = results_df["param_knn__n_neighbors"]
    results_df["mean_score"] = results_df["mean_test_score"]

    fig_cv = px.line(results_df, x="k", y="mean_score",
        title=f"2a. Cross-Validated Negative MSE vs. K (Best K = {best_k})", markers=True, 
        labels={"k": "Number of Neighbors (k)", "mean_score": "Mean CV Negative MSE"}, height=450)

    # Final Model Evaluation
    knn_best = grid.best_estimator_
    y_pred_best = knn_best.predict(X_test)
    
    # 1. Metrics Table
    metrics_df, rmse_test = calculate_metrics(y_test, y_pred_best, best_k)
    
    # 3. Predicted vs. Actual Scatter Plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))

    scatter = ax_scatter.scatter(y_test, y_pred_best, 
                                 c=X_test['fire_brightness'], cmap='plasma', alpha=0.7, s=70, edgecolor='k')
    
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax_scatter.set_xlabel("Actual AQI_PM25", fontsize=12)
    ax_scatter.set_ylabel("Predicted AQI_PM25", fontsize=12)
    ax_scatter.set_title("2c. Predicted vs Actual AQI_PM25", fontsize=14)
    
    fig_scatter.colorbar(scatter).set_label('Fire Brightness (Predictor)', fontsize=12)
    plt.close(fig_scatter)

    # Calculate median RMSE from cross-validation
    median_rmse = np.sqrt(-grid.best_score_)
    
    return metrics_df, fig_cv, fig_scatter, fig_corr, best_k, median_rmse

# ==============================================================================
# 3. PRINCIPAL COMPONENT ANALYSIS (From pca_streamlit.py)
# ==============================================================================

def run_pca_analysis(data: pd.DataFrame) -> Tuple[px.line, go.Figure, plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:
    # Analysis logic copied directly from pca_streamlit.py
    df = data.copy()
    
    pca_features = [
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'wind_direction_10m_dominant',
        'precipitation_sum', 'precipitation_hours', 'et0_fao_evapotranspiration',
        'distance_to_fire_km', 'fire_brightness', 'fire_frp', 'fires_within_50km',
        'fires_within_100km'
    ]

    X = df[pca_features].dropna()
    df_aligned = df.loc[X.index].copy()

    pca_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA())])
    pca_pipe.fit(X)

    pca = pca_pipe.named_steps['pca']
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_components = len(explained_var)
    
    # 1. Scree Plot
    ev_df = pd.DataFrame({"PC": np.arange(1, n_components + 1), "ExplainedVariance": explained_var,
                          "CumulativeVariance": cumulative_var})

    fig_scree = px.line(ev_df, x="PC", y="ExplainedVariance", markers=True,
                        title="3a. Scree Plot: Proportion of Variance Explained", height=450, 
                        labels={"PC": "Principal Component", "ExplainedVariance": "Explained Variance Ratio"})
    fig_scree.add_scatter(x=ev_df["PC"], y=[1/n_components] * n_components, mode="lines", 
                          name="Average Variance (Kaiser rule)", line=dict(dash='dash', color='gray'))

    # 5. Cumulative Variance Summary Table
    var_summary_df = pd.DataFrame({
        'PC Count': [1, 3, 5, 6, n_components],
        'Cumulative Variance (%)': [cumulative_var[i-1] * 100 for i in [1, 3, 5, 6, n_components]]
    }).set_index('PC Count').round(1)
    
    # PCA Scores and Loadings
    X_pca_scores = pca_pipe.transform(X)
    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores_df = pd.DataFrame(X_pca_scores, columns=pc_cols, index=X.index)
    scores_df['PM25'] = df_aligned['PM25']
    
    loadings = pca.components_.T
    loading_df_full = pd.DataFrame(loadings, index=pca_features, columns=pc_cols)
    loading_df_2d = loading_df_full[['PC1', 'PC2']] # For biplot and corr circle

    # 2. Biplot (PC1 vs PC2 with Loadings) - Scatter plot slightly larger
    arrow_scale = 3 

    fig_biplot = go.Figure()
    fig_biplot.add_trace(go.Scatter( 
        x=scores_df['PC1'], y=scores_df['PC2'], mode='markers',
        marker=dict(size=5, color=scores_df['PM25'], colorscale='RdYlGn_r', opacity=0.6,
                    colorbar=dict(title='PM2.5 (µg/m³)')), showlegend=False,
        hovertemplate='PM2.5: %{marker.color:.2f}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))

    for var_name, row in loading_df_2d.iterrows():
        x_arrow, y_arrow = row['PC1'] * arrow_scale, row['PC2'] * arrow_scale
        fig_biplot.add_trace(go.Scatter(x=[0, x_arrow], y=[0, y_arrow], mode='lines', 
                                        line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
        fig_biplot.add_annotation(x=x_arrow * 1.15, y=y_arrow * 1.15, text=f'<b>{var_name}</b>', 
                                  showarrow=False, font=dict(size=9, color='darkred'),
                                  bgcolor='rgba(255,255,255,0.7)', borderwidth=1)

    # Biplot size adjustment: slightly larger than (400, 500)
    fig_biplot.update_layout(title=f'3b. PCA Biplot: PC1 ({explained_var[0]*100:.1f}%) vs PC2 ({explained_var[1]*100:.1f}%)',
                             height=650, width=750)
    
    # 3. Correlation Circle (PC1 vs PC2 Loadings Plot) - Keep smaller size
    fig_corr_circle, ax_corr_circle = plt.subplots(figsize=(6, 6))
    loadings_df = loading_df_2d.copy()

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=2)
    ax_corr_circle.add_patch(circle)

    # Plot loadings as arrows
    for i, feature in enumerate(pca_features):
        x = loadings_df.loc[feature, 'PC1']
        y = loadings_df.loc[feature, 'PC2']
        
        ax_corr_circle.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                             fc='red', ec='red', linewidth=2, alpha=0.7)
        
        # Label
        ax_corr_circle.text(x * 1.15, y * 1.15, feature, fontsize=8, 
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Axes
    ax_corr_circle.axhline(0, color='black', linewidth=0.5)
    ax_corr_circle.axvline(0, color='black', linewidth=0.5)

    ax_corr_circle.set_xlim(-1.2, 1.2)
    ax_corr_circle.set_ylim(-1.2, 1.2)
    ax_corr_circle.set_aspect('equal')
    ax_corr_circle.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=10)
    ax_corr_circle.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
    ax_corr_circle.set_title('3c. Correlation Circle: Variable Contributions to PC1 & PC2', 
                 fontsize=12, fontweight='bold')
    ax_corr_circle.grid(alpha=0.3)
    plt.tight_layout()
    plt.close(fig_corr_circle) # Close the figure to manage memory
    
    # 4. Loadings Table (PC1, PC2) - Still needed for MLR analysis, but removed from display
    loading_df_summary = loading_df_full[['PC1', 'PC2']].round(3)
    loading_df_summary.columns = [f'PC1 ({explained_var[0]*100:.1f}%)', f'PC2 ({explained_var[1]*100:.1f}%)']

    # 4. Loadings Heatmap (Matplotlib) - Make smaller
    n_pcs_show = 5
    loadings_df_heatmap = loading_df_full.iloc[:, :n_pcs_show]

    # Heatmap size adjustment: smaller
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(6, 4))
    sns.heatmap(loadings_df_heatmap, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Loading Value'}, ax=ax_heatmap)

    ax_heatmap.set_title('3d. Feature Loadings on Principal Components (PC1-PC5)', fontsize=14, pad=35)
    variance_text = ' | '.join([f'PC{i+1}: {explained_var[i]*100:.1f}%' for i in range(n_pcs_show)])
    ax_heatmap.text(0.5, 1.05, f'Variance Explained: {variance_text}', transform=ax_heatmap.transAxes, 
            ha='center', fontsize=10, style='italic')
    plt.close(fig_heatmap)

    return fig_scree, fig_biplot, fig_corr_circle, fig_heatmap, loading_df_summary, var_summary_df


# ==============================================================================
# 4. MULTIPLE LINEAR REGRESSION (From mlr_streamlit.py)
# ==============================================================================

def run_mlr_analysis(data: pd.DataFrame) -> Tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DataFrame, px.scatter]:
    # Analysis logic copied directly from mlr_streamlit.py
    df = data.copy()
    
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5
    
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        # Handle the 'const' column (if present) gracefully
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data

    def boot(X, y, S=1000):
        coef_store = np.zeros((S, X.shape[1]))
        n = len(X)
        y_df = pd.DataFrame(y, columns=['PM25'])

        for _ in range(S):
            idx = np.random.choice(n, n, replace=True)
            X_b = X.iloc[idx].reset_index(drop=True)
            y_b = y_df.iloc[idx].reset_index(drop=True)
            boot_model = sm.OLS(y_b, X_b).fit()
            coef_store[_] = boot_model.params

        boot_results = pd.DataFrame(coef_store, columns=X.columns)
        return boot_results.mean(), boot_results.std()
    
    # Feature Preparation
    # NOTE: The MLR interpretations mention removing 'fire_frp' and other VIF-related columns. 
    X_final_cols = ['relative_humidity_2m_mean', 'et0_fao_evapotranspiration', 
                    'latitude', 'longitude', 'temperature_2m_mean', 
                    'wind_speed_10m_mean', 'precipitation_sum', 
                    'fires_within_50km', 'fires_within_100km', 
                    'distance_to_fire_km', 'fire_brightness']
    target = 'PM25'

    df[X_final_cols] = df[X_final_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=X_final_cols + [target], inplace=True)
    
    y_df = df[[target]].copy().reset_index(drop=True)
    
    scaler = StandardScaler()
    scaled_X_array = scaler.fit_transform(df[X_final_cols])
    scaled_df = pd.DataFrame(scaled_X_array, columns=X_final_cols).reset_index(drop=True)

    X_full = sm.add_constant(scaled_df)
    y_full = y_df[[target]].to_numpy(dtype='float64')
    
    # 1. Final VIF Table
    final_vif_all = calculate_vif(X_full) 
    
    X = X_full.copy()
    y = pd.DataFrame(y_full, columns=['PM25'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42) 

    # 2. OLS Model Training
    model_train = sm.OLS(y_train, X_train).fit()

    # Bootstrapping
    mean_coef, std_coef = boot(X_train, y_train, 1000)

    # 3. Bootstrap vs OLS SE Comparison
    ols_se = model_train.bse
    comparison_df = pd.DataFrame({
        'Bootstrap SE': std_coef,
        'OLS SE': ols_se,
        'Abs Diff': np.abs(std_coef - ols_se)
    })
    
    # 4. Performance Metrics Table
    yhat_tr = model_train.predict(X_train)
    yhat_val = model_train.predict(X_val)
    yhat_test = model_train.predict(X_test)
    
    performance_metrics = {
        'Train': [rmse(y_train, yhat_tr), r2_score(y_train, yhat_tr)],
        'Validate': [rmse(y_val, yhat_val), r2_score(y_val, yhat_val)],
        'Test': [rmse(y_test, yhat_test), r2_score(y_test, yhat_test)]
    }
    performance_df = pd.DataFrame(performance_metrics, index=['RMSE', 'R²']).T.round(4)
    
    # 5. Residuals vs Fitted Plot (Using Train Model on Test Data)
    plot_df = pd.DataFrame({
        "Actual_PM25": y_test['PM25'].values,
        "Fitted_PM25": yhat_test,
        "Residuals": y_test['PM25'].values - yhat_test
    })
    
    fig_resid = px.scatter(
        plot_df, x="Fitted_PM25", y="Residuals", 
        title="4c. Residuals vs Fitted Values (Using Test Data)",
        labels={"Fitted_PM25": "Predicted PM25", "Residuals": "Residual Error"},
        opacity=0.6, height=500
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")

    return final_vif_all, model_train, comparison_df, performance_df, fig_resid


# ==============================================================================
# 5. LOGISTIC REGRESSION (From logit_streamlit.py)
# ==============================================================================

def run_logistic_model(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, go.Figure, float]:
    # Analysis logic copied directly from logit_streamlit.py
    df = data.copy()
    
    # Corrected X_cols based on previous interaction (assuming original data had full names)
    X_cols = ['relative_humidity_2m_mean', 'et0_fao_evapotranspiration', 
              'latitude', 'longitude', 'temperature_2m_mean', 
              'wind_speed_10m_mean', 'precipitation_sum', 
              'fires_within_50km', 'fires_within_100km', 
              'distance_to_fire_km', 'fire_brightness']

    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['PM25'] + X_cols, inplace=True)
    
    y_median = df['PM25'].median()
    y = (df['PM25'] > y_median).astype(int) # 0 = Low/At Median, 1 = Above Median (High Risk)
    X = df[X_cols]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42) 

    pipe = Pipeline(steps = [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)
    
    # 1. Metrics Table
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_score)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Log Loss', 'Threshold'],
        'Value': [f"{accuracy:.4f}", f"{logloss:.4f}", f"{y_median:.2f} (PM25)"]
    }).set_index('Metric')

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual Low PM25 (0)', 'Actual High PM25 (1)'], 
        columns=['Predicted Low PM25 (0)', 'Predicted High PM25 (1)']
    )

    # 3. & 4. ROC Curve and AUC
    y_score_positive = y_score[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score_positive)
    auc_score = roc_auc_score(y_test, y_score_positive)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {auc_score:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guessing'))
    fig_roc.update_layout(
        title='5c. Receiver Operating Characteristic (ROC) Curve', 
        xaxis_title='False Positive Rate', 
        yaxis_title='True Positive Rate',
        height=450
    )
    
    return metrics_df, cm_df, fig_roc, auc_score


# ==============================================================================
# STREAMLIT DISPLAY FUNCTIONS
# ==============================================================================

def display_kmeans_tab(df: pd.DataFrame):
    st.header("1. K-Means Clustering: Unsupervised Grouping")
    st.markdown(
        """
        K-Means clustering is utilized to discover natural, unlabelled groupings, or environmental regimes, in the fire and weather data.
        """
    )
    
    with st.spinner("Running K-Means Clustering Analysis..."):
        fig_elbow, fig_sil, centers_df, pm25_means = run_kmeans_analysis(df)
        
    st.subheader("1. Optimal Cluster Selection")
    
    # Row 1: Elbow Method
    st.markdown("##### 1a. Elbow Method (WCSS)")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.info("The **steep decrease in WCSS from K=1 to K=3** (the 'elbow') suggests that $K=2$ to $K=5$ clusters potentially capture most of the patterns in the data.")
        
    # Row 2: Silhouette Score
    st.markdown("##### 1b. Silhouette Score")
    st.plotly_chart(fig_sil, use_container_width=True)
    st.info(f"The highest Silhouette Score, a very high **0.88** for **$K=2$**, indicates clusters that are theoretically well-separated. Scores for $K \ge 3$ are less than 0.25, suggesting weak or overlapping structures.")
    
    st.subheader("2. Cluster Characteristics (K=2)")
    
    # Row 3: Cluster Centers (Table)
    st.markdown("##### 2a. Scaled Feature Centers")
    st.dataframe(centers_df, use_container_width=True)
    st.caption("Values are standardized (mean=0, std=1). Positive values indicate above-average feature levels for that cluster.")
    st.markdown(
        """
        * **Cluster 1** (The high-PM2.5 cluster) is associated with **higher temperature** and **more nearby fires**.
        * **Cluster 0** (The low-PM2.5 cluster) is associated with **higher wind speed**, **greater precipitation**, and **longer distances from fires**.
        """
    )
        
    # Row 4: PM25 Mean (Table)
    st.markdown("##### 2b. Average PM25 per Cluster")
    st.dataframe(pm25_means.to_frame(), use_container_width=False)
    st.success(
        f"**Cluster 1** (PM25: **{pm25_means.iloc[1]:.2f}**) captures **extreme pollution events** (only 2 points), while **Cluster 0** (PM25: **{pm25_means.iloc[0]:.2f}**) captures **normal air quality** (almost the entire dataset)."
    )
    st.warning(
        """
        **Overall Conclusion:** The clustering is **highly unbalanced**. The K-Means model is largely grouping the majority of the data together while isolating extreme outliers, indicating the model is not finding meaningful, balanced subgroups. This suggests that most of the data are within average environmental conditions, and the model is unable to find meaningful patterns beyond extreme cases.
        """
    )
    st.markdown("---")


def display_knn_tab(df: pd.DataFrame):
    st.header("2. K-Nearest Neighbors Regression: Predicting AQI")
    st.markdown(
        """
        KNN regression is employed to predict AQI (PM2.5) by using the fire, weather, and location datasets. KNN operates by measuring environmental similarity and averaging the AQI values of the K closest neighbors.
        """
    )
    with st.spinner("Running KNN Regression and Hyperparameter Tuning..."):
        metrics_df, fig_cv, fig_scatter, fig_corr, best_k, median_rmse = run_knn_regression(df)

    st.subheader("1. Model Tuning")
    
    # Row 1: CV Plot vs K
    st.markdown("##### 2a. Cross-Validation for Optimal K")
    st.plotly_chart(fig_cv, use_container_width=True)
    st.info(f"Optimal $k$ is **{best_k}**, chosen to minimize prediction error (Negative MSE). This means each prediction uses the **{best_k} closest points**.")

    st.subheader("2. Key Metrics")
    
    # Row 2: Metrics Table and Interpretation
    st.markdown("##### 2b. Key Metrics on Test Set")
    st.dataframe(metrics_df, use_container_width=True)
    st.success(f"The R² score of **~51%** means the model explains approximately 51% of the variation in AQI, suggesting that other factors (e.g., traffic, industrial emissions) contribute significantly to the remaining 53% of unexplained variation.")
    st.markdown(f"The cross-validation RMSE of **$\pm{median_rmse:.2f}$ AQI units** means the model's predictions are, on average, off by about 13 AQI units.")

    st.subheader("3. Predicted vs. Actual Performance & Diagnostics")

    # Row 3: Scatter Plot vs Underprediction Analysis
    st.markdown("##### 2c. Predicted vs Actual AQI_PM25")
    st.pyplot(fig_scatter)
    st.warning(
        """
        **Underprediction at Higher Values:** As actual AQI increases beyond approximately 75, the predicted values start falling below the perfect prediction line. This indicates the model **underestimates high AQI events**, which is a common effect of KNN's smoothing (averaging) nature on extreme outliers.
        * The dense cluster at lower AQI shows the model is highly accurate at predicting typical/low AQI values.
        """
    )
        
    # Row 4: Correlation Heatmap
    st.markdown("##### 2d. Predictor Correlation Heatmap")
    st.pyplot(fig_corr)
    st.caption(
        """
        Multicollinearity, while not breaking the KNN model, was minimized by removing redundant variables like 'Fire FRP' (strongly correlated with 'fire brightness') to increase computational efficiency.
        """
    )
    st.markdown("---")


def display_pca_tab(df: pd.DataFrame):
    st.header("3. Principal Component Analysis (PCA): Feature Structure")
    st.markdown(
        """
        PCA reduces 14 weather/fire features to interpret the data's underlying structure and its relation to PM2.5.
        """
    )
    with st.spinner("Running PCA..."):
        fig_scree, fig_biplot, fig_corr_circle, fig_heatmap, loading_df_sum, var_sum_df = run_pca_analysis(df)
    
    st.subheader("1. Variance Explained")

    # Row 1: Scree Plot
    st.markdown("##### 3a. Scree Plot: Proportion of Variance Explained")
    st.plotly_chart(fig_scree, use_container_width=True)
    st.info("The scree plot indicates a diminishing return after PC3, suggesting the majority of the data's complexity is captured by the first few components.")
    
    # Row 2: Cumulative Variance Table
    st.markdown("##### Cumulative Variance Summary")
    st.dataframe(var_sum_df, use_container_width=False)
    st.success(f"The first **6** PCs capture **{var_sum_df.loc[6, 'Cumulative Variance (%)']:.1f}%** of the total variance.")
    
    st.subheader("2. Biplot and Correlation Circle (PC1 vs. PC2)")
    
    # Row 3: Biplot
    st.markdown("##### 3b. PCA Biplot (Scores + Loadings)")
    # use_container_width=False to respect the fixed smaller width (width=550)
    st.plotly_chart(fig_biplot, use_container_width=False) 
    st.info(
        "High PM2.5 scores cluster along the positive **PC1 axis**, confirming that this factor is the primary environmental driver of poor air quality."
    )

    # Row 4: Correlation Circle
    st.markdown("##### 3c. Correlation Circle: Variable Contributions to PC1 & PC2")
    st.pyplot(fig_corr_circle)
    st.caption(
        "Variables pointing in the same direction (e.g., Fire Brightness and Temperature) are positively correlated. Variables near the circle are better represented by this 2D plane."
    )

    st.subheader("3. Component Interpretation (Loadings)")
    
    # Row 5: Loadings Heatmap
    st.markdown("##### 3d. Feature Loadings on Principal Components (PC1-PC5)")
    st.pyplot(fig_heatmap)
    
    # The user requested to remove the "Top Loadings (PC1 and PC2)" table here.
    
    # Detailed PCA Interpretations
    st.markdown("##### Detailed PCA Component Interpretations")
    st.markdown(
        """
        * **PC1 (27.9% variance explained): Thermal Axis**
            * Separates hot, dry conditions (high scores) from cool, humid conditions (low scores).
            * This is the largest source of variation, as temperature varies dramatically across seasons and locations.
            * Fire variables do not load heavily on PC1, suggesting fires occur across various temperature ranges.
        * **PC2 (15.6% variance explained): Precipitation & Moisture**
            * Separates rainy days (high scores: higher precipitation, humidity, and min temperature) from dry days (low scores: dry conditions, clearer skies).
        * **PC3 (11.7% variance explained): Fire Proximity & Activity**
            * Captures the presence of fires near monitoring sites (within 50km and 100km radius of a site).
        * **PC4 (10.4% variance explained): Fire Intensity**
            * Captures the brightness and radiative power of fires.
        * **PC5 (8.1% variance explained): Wind Patterns**
            * Captures strong winds from a particular direction, with lower humidity.
        
        The separation of **fire characteristics (PC3, PC4)** from **weather conditions (PC1, PC2, PC5)** demonstrates that wildfire activity, while dependent on weather, also varies *independently* of these basic weather patterns, likely driven by factors like fuel type, ignition, and historical fire management.
        """
    )
    st.markdown("---")


def display_mlr_tab(df: pd.DataFrame):
    st.header("4. Multiple Linear Regression (OLS): PM2.5 Prediction")
    st.markdown(
        """
        An OLS model predicts continuous PM2.5 levels using standardized features. Features were selected to mitigate multicollinearity (VIF), and the model's stability was verified using bootstrapping.
        """
    )
    with st.spinner("Running MLR with VIF check and Bootstrapping..."):
        vif_df, ols_summary, boot_compare_df, performance_df, fig_resid = run_mlr_analysis(df)
    
    st.subheader("1. Model Summary and Diagnostics")

    # Row 1: OLS Summary (Code block)
    st.markdown("##### 4a. OLS Model Summary (Training Data)")
    st.code(ols_summary.summary().as_text(), language='text')

    # Row 2: VIF Table
    st.markdown("##### Multicollinearity Check (VIF)")
    st.dataframe(vif_df, use_container_width=True)
    st.warning(
        """
        **VIF Interpretation:** Variables with moderate to strong multicollinearity (VIF > 5), such as 'Fire FRP' (correlated with 'Fire Brightness'), were removed to ensure stable coefficient estimation.
        """
    )

    # Row 3: Performance Metrics Table
    st.markdown("##### 4b. Performance Metrics (Train/Validate/Test)")
    st.dataframe(performance_df, use_container_width=True)
    st.success(
        "The low difference between Train, Validate, and Test RMSE and R² values confirms the model's **robustness** and good generalization to new data."
    )
    
    st.subheader("2. Model Evaluation and Assumptions")
    
    # Row 4: Residuals Plot and Analysis
    st.markdown("##### 4c. Residuals vs Fitted Plot")
    st.plotly_chart(fig_resid, use_container_width=True)
    st.warning(
        """
        **Flaw (Heteroscedasticity):** The increasing spread of residuals as fitted values increase, along with the large number of high positive residuals (up to 80), indicates that the model is **less accurate at predicting high PM2.5 values** and generally **underpredicts** the true value in these cases.
        """
    )

    # Row 5: Bootstrap SE Comparison and Interpretation
    st.markdown("##### Bootstrap SE Comparison")
    st.dataframe(boot_compare_df[['OLS SE', 'Bootstrap SE']], use_container_width=True)
    st.info(
        "The similarity between OLS Standard Errors (SE) and Bootstrapped SEs indicates that **OLS assumptions are largely met**, and traditional statistical inference (p-values) from the OLS summary is considered reliable."
    )
        
    st.subheader("3. Conclusion")
    
    # Calculate scaled range for interpretation
    pm25_range = df['PM25'].max() - df['PM25'].min()
    rmse_test = performance_df.loc['Test', 'RMSE']
    
    st.markdown(
        f"""
        The low R² score indicates a poor model fit for prediction, despite all predictors being statistically significant. 
        
        However, the RMSE of **~${rmse_test:.2f}$** means predictions are off by about **${rmse_test / pm25_range * 100:.1f}\%$** of the total PM2.5 range (on a range of $\sim {pm25_range:.1f}$ $\mu$g/m³). This suggests **middling results** for typical cases, but confirms the model is unreliable for predicting extreme pollution events.
        """
    )
    st.markdown("---")


def display_logistic_tab(df: pd.DataFrame):
    st.header("5. Logistic Regression: High PM2.5 Risk Classification")
    st.markdown(
        """
        This model classifies air quality as **'High Risk'** (PM2.5 **above** the median) 
        or **'Low Risk'** (PM2.5 **at or below** the median) based on scaled features.
        """
    )
    with st.spinner("Running Logistic Regression..."):
        metrics_df, cm_df, fig_roc, auc_score = run_logistic_model(df)

    st.subheader("1. Key Metrics and Confusion Matrix")
    
    # Row 1: Key Metrics Table
    st.markdown("##### 5a. Key Metrics")
    st.dataframe(metrics_df, use_container_width=True)
    st.warning(
        f"""
        The overall accuracy of **~67%** is acceptable. However, the high **Log Loss of {metrics_df.loc['Log Loss', 'Value']}** indicates a flaw: the model's predicted probabilities are unreliable, meaning it is often highly confident in predictions that turn out to be incorrect.
        """
    )
        
    # Row 2: Confusion Matrix Table
    st.markdown("##### 5b. Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)
    st.info(
        "The matrix shows roughly twice as many true positives and true negatives as false positives and false negatives, indicating a reasonable balance in classification errors."
    )
            
    st.subheader("2. ROC Curve and Discrimination")
    
    # Row 3: ROC Curve and AUC Analysis
    st.markdown("##### 5c. Receiver Operating Characteristic (ROC) Curve")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.success(
        f"The **Area Under the Curve (AUC)** is **{auc_score:.3f}**. This indicates that the model has **good discrimination** between high and low PM2.5 risk, with a 72% chance of correctly distinguishing between the classes."
    )
    st.caption("The ROC curve shows the trade-off between the True Positive Rate and the False Positive Rate. A curve that rises slowly (as seen here) indicates that discrimination ability is only slightly better than random guessing in some regions of probability.")
    st.markdown("---")


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Unified Wildfire Analysis")
    st.title("Unified Environmental Impact Analysis on Air Quality")
    st.markdown(
        """
        This dashboard consolidates five separate analyses (Clustering, KNN, PCA, MLR, and Logit) 
        into a single Streamlit application to model and understand the impact of wildfire and 
        environmental factors on air quality (PM2.5/AQI).
        """
    )
    
    # Load data for non-PCA models (index_col=None)
    df_no_index = load_data(index_col=None) 
    # Load data for PCA (index_col=0, as required by pca_streamlit.py)
    df_pca = load_data(index_col=0) 

    if df_no_index is None or df_pca is None:
        st.error("Cannot proceed without loading all necessary data files.")
        return

    # Create five tabs
    tab_kmeans, tab_knn, tab_pca, tab_mlr, tab_logit = st.tabs([
        "1. K-Means Clustering", 
        "2. KNN Regression", 
        "3. PCA (Feature Structure)", 
        "4. MLR (PM2.5 Prediction)", 
        "5. Logistic Regression (Risk)"
    ])

    with tab_kmeans:
        display_kmeans_tab(df_no_index.copy())
        
    with tab_knn:
        display_knn_tab(df_no_index.copy())

    with tab_pca:
        display_pca_tab(df_pca.copy()) 

    with tab_mlr:
        display_mlr_tab(df_no_index.copy())

    with tab_logit:
        display_logistic_tab(df_no_index.copy())

    st.markdown("---")

if __name__ == '__main__':
    main()