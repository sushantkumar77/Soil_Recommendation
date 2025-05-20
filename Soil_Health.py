import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_regression, RFECV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from scipy import stats
from sklearn.inspection import permutation_importance
import itertools
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess your cleaned data
# Replace with your actual path to the cleaned CSV
data = pd.read_csv(r'Maharashtra_all_bands.csv')

# Select relevant columns and calculate indices (as in your original code)
final_data = data[['date', 'village', 'geometry', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                   'B8A', 'B9', 'B11', 'B12', 'N', 'P', 'K', 'OC', 'pH']]

# Scale spectral bands
scale_factor = 10000.0
band_columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
for band in band_columns:
    final_data[band] = final_data[band] / scale_factor

# Calculate vegetation indices
# Basic indices
final_data['NDVI'] = (final_data['B8'] - final_data['B4']) / (final_data['B8'] + final_data['B4'])
final_data['GNDVI'] = (final_data['B8'] - final_data['B3']) / (final_data['B8'] + final_data['B3'])
final_data['SAVI'] = ((final_data['B8'] - final_data['B4']) / (final_data['B8'] + final_data['B4'] + 0.5)) * (1 + 0.5)
final_data['SR'] = final_data['B8'] / final_data['B4']
final_data['IR_G'] = final_data['B8'] / final_data['B3']
final_data['EVI'] = (final_data['B8'] - final_data['B4']) / (
            final_data['B8'] + 6 * final_data['B4'] - 7.5 * final_data['B2'] + 1)
final_data['CVI'] = final_data['B8'] * (final_data['B4'] / final_data['B3'] ** 2)
final_data['GLI'] = (2 * final_data['B3'] - final_data['B4'] - final_data['B2']) / (
            2 * final_data['B3'] + final_data['B4'] + final_data['B2'])
final_data['NDWI'] = (final_data['B3'] - final_data['B8']) / (final_data['B3'] + final_data['B8'])
final_data['RECl'] = (final_data['B8'] / final_data['B4']) - 1
final_data['OSAVI'] = (final_data['B8'] - final_data['B4']) / (final_data['B8'] + final_data['B4'] + 0.16)
final_data['ARVI'] = (final_data['B8'] - (2 * final_data['B4']) + final_data['B2']) / (
            final_data['B8'] + (2 * final_data['B4']) + final_data['B2'])
final_data['VARI'] = (final_data['B3'] - final_data['B4']) / (final_data['B4'] + final_data['B3'] - final_data['B2'])
final_data['GCI'] = final_data['B8'] / (final_data['B3'] - 1)
final_data['SIPI'] = (final_data['B8'] - final_data['B2']) / (final_data['B8'] - final_data['B4'])
final_data['NDBI'] = (final_data['B11'] - final_data['B8']) / (final_data['B11'] + final_data['B8'])
final_data['MNDWI'] = (final_data['B3'] - final_data['B11']) / (final_data['B3'] + final_data['B11'])
final_data['SR_n2'] = final_data['B8'] / final_data['B3']
final_data['SR_N'] = final_data['B8'] / final_data['B2']
final_data['TBVI1'] = final_data['B5'] / (final_data['B5'] + final_data['B2'])

# Additional N-sensitive vegetation indices
final_data['BNDVI'] = (final_data['B8'] - final_data['B2']) / (final_data['B8'] + final_data['B2'])  # Blue NDVI
final_data['RENDVI'] = (final_data['B8A'] - final_data['B5']) / (final_data['B8A'] + final_data['B5'])  # Red-Edge NDVI
final_data['MCARI'] = ((final_data['B5'] - final_data['B4']) - 0.2 * (final_data['B5'] - final_data['B3'])) * (
            final_data['B5'] / final_data['B4'])  # Modified Chlorophyll Absorption Ratio Index
final_data['MTCI'] = (final_data['B6'] - final_data['B5']) / (
            final_data['B5'] - final_data['B4'])  # MERIS Terrestrial Chlorophyll Index
final_data['IRECI'] = (final_data['B7'] - final_data['B4']) / (
            final_data['B5'] / final_data['B6'])  # Inverted Red-Edge Chlorophyll Index
final_data['CRE'] = final_data['B6'] / final_data['B5'] - 1  # Chlorophyll Red-Edge index
final_data['MSR'] = (final_data['B8'] / final_data['B4'] - 1) / (
            np.sqrt(final_data['B8'] / final_data['B4']) + 1)  # Modified Simple Ratio

# Extract coordinates
final_data['latitude'] = final_data['geometry'].apply(lambda x: float(x.split(' ')[2].strip(')')))
final_data['longitude'] = final_data['geometry'].apply(lambda x: float(x.split(' ')[1].strip('(')))

# Add ID column
final_data['id'] = range(1, len(final_data) + 1)

# Clean data
# Select numeric columns
numeric_data = final_data.select_dtypes(include=[np.number])

# Handle infinite values and NaN
clean_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()

# Verify we don't have any NaN or Inf values
print("Dataset shape after cleaning:", clean_data.shape)
print("NaN values:", clean_data.isna().sum().sum())
print("Infinity values:", np.isinf(clean_data).sum().sum())

# Select features and target for N prediction
# Excluding other soil properties that wouldn't be available during prediction
X = clean_data.drop(['id', 'N', 'P', 'K', 'OC', 'pH'], axis=1)
y = clean_data['N']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Examine target distribution
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of Nitrogen (N) Values')
plt.show()

# If target is skewed, consider transformation
skewness = stats.skew(y)
print(f"Skewness of N: {skewness}")

# Apply log transformation if highly skewed (skewness > 1 or < -1)
if abs(skewness) > 1:
    y_transformed = np.log1p(y) if skewness > 0 else -np.log1p(-y)
    print("Applied log transformation due to skewness")
    use_transformed_y = True
else:
    y_transformed = y.copy()
    use_transformed_y = False

# Feature selection based on correlation with target
correlation = X.corrwith(y)
top_features = correlation.abs().sort_values(ascending=False)
print("\nTop 15 features by correlation with N:")
print(top_features.head(15))

# Optional: Plot correlation with target
plt.figure(figsize=(12, 8))
correlation.abs().sort_values().plot(kind='barh')
plt.title('Feature Correlation with Nitrogen (N)')
plt.tight_layout()
plt.show()

# Optional: Feature importance-based selection using Random Forest
feature_selector = SelectFromModel(
    RandomForestRegressor(n_estimators=100, random_state=42),
    max_features=15
)
feature_selector.fit(X, y)
selected_mask = feature_selector.get_support()
selected_features = X.columns[selected_mask]
print("\nSelected features by importance:")
print(selected_features.tolist())

# Set up training and testing data for the advanced model
X_selected = X[selected_features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y_transformed if use_transformed_y else y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = RobustScaler()  # More robust against outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------- BAND-SPECIFIC ANALYSIS FOR NITROGEN PREDICTION -------

# Function to evaluate prediction accuracy using only specific features
def evaluate_features(features, X_train, X_test, y_train, y_test, model_type=RandomForestRegressor, name="Features"):
    """Evaluate prediction accuracy using only the specified features"""

    # Select only the specified features
    X_train_selected = X_train[:, [list(X_train_columns).index(feat) for feat in features if feat in X_train_columns]]
    X_test_selected = X_test[:, [list(X_test_columns).index(feat) for feat in features if feat in X_test_columns]]

    # Train the model
    model = model_type(random_state=42)
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_test_pred = model.predict(X_test_selected)

    # Calculate metrics
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    return {
        'name': name,
        'features': features,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_count': len(features)
    }


# ----- BAND ANALYSIS FOR NITROGEN PREDICTION -----
print("\n" + "=" * 50)
print("ANALYSIS OF BAND-SPECIFIC PREDICTION ACCURACY FOR NITROGEN")
print("=" * 50)

# Store column names for reference
X_train_columns = X_train.columns if hasattr(X_train, 'columns') else selected_features
X_test_columns = X_test.columns if hasattr(X_test, 'columns') else selected_features

# 1. Individual raw band evaluation
raw_bands = [band for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
             if band in X_train_columns]
print(f"\nEvaluating {len(raw_bands)} individual spectral bands...")

band_results = []
for band in raw_bands:
    if band in X_train_columns:
        result = evaluate_features([band], X_train_scaled, X_test_scaled, y_train, y_test,
                                   model_type=RandomForestRegressor, name=f"Band {band}")
        band_results.append(result)
        print(f"Band {band}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")

# 2. Band combinations (Red + NIR, etc.)
key_band_combinations = [
    (['B4', 'B8'], "Red + NIR"),  # Basic vegetation response
    (['B3', 'B4', 'B8'], "Green + Red + NIR"),  # Extended vegetation
    (['B5', 'B6', 'B7'], "Red Edge Bands"),  # Red edge response sensitive to N
    (['B2', 'B3', 'B4', 'B8'], "Vis + NIR"),  # Full visible spectrum + NIR
    (['B8', 'B11', 'B12'], "NIR + SWIR"),  # Water and soil minerals
]

print("\nEvaluating key band combinations...")
band_combo_results = []
for bands, name in key_band_combinations:
    valid_bands = [band for band in bands if band in X_train_columns]
    if len(valid_bands) >= 2:  # Ensure we have at least 2 bands
        result = evaluate_features(valid_bands, X_train_scaled, X_test_scaled, y_train, y_test,
                                   model_type=RandomForestRegressor, name=name)
        band_combo_results.append(result)
        print(f"{name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")

# 3. All raw bands together
all_bands_result = evaluate_features(raw_bands, X_train_scaled, X_test_scaled, y_train, y_test,
                                     model_type=RandomForestRegressor, name="All Raw Bands")
print(f"\nAll Raw Bands: R² = {all_bands_result['r2']:.4f}, RMSE = {all_bands_result['rmse']:.4f}")

# 4. Vegetation indices without raw bands
veg_indices = [col for col in X_train_columns if col not in raw_bands + ['latitude', 'longitude']]
veg_indices_result = evaluate_features(veg_indices, X_train_scaled, X_test_scaled, y_train, y_test,
                                       model_type=RandomForestRegressor, name="All Vegetation Indices")
print(f"All Vegetation Indices: R² = {veg_indices_result['r2']:.4f}, RMSE = {veg_indices_result['rmse']:.4f}")

# 5. Complete model (raw bands + indices)
all_features_result = evaluate_features(X_train_columns, X_train_scaled, X_test_scaled, y_train, y_test,
                                        model_type=RandomForestRegressor, name="All Features")
print(f"All Features (Bands + Indices): R² = {all_features_result['r2']:.4f}, RMSE = {all_features_result['rmse']:.4f}")

# Collect all results for comparison
all_results = band_results + band_combo_results + [all_bands_result, veg_indices_result, all_features_result]

# Create comparison plots
plt.figure(figsize=(14, 8))
all_names = [r['name'] for r in all_results]
all_r2 = [r['r2'] for r in all_results]

# Sort by R² scores
sorted_indices = np.argsort(all_r2)
sorted_names = [all_names[i] for i in sorted_indices]
sorted_r2 = [all_r2[i] for i in sorted_indices]

# Create horizontal bar chart
plt.barh(range(len(sorted_names)), sorted_r2, align='center')
plt.yticks(range(len(sorted_names)), sorted_names)
plt.xlabel('R² Score')
plt.title('Nitrogen Prediction Accuracy by Band/Feature Groups')
plt.grid(True, linestyle='--', alpha=0.7)

# Add R² values at the end of each bar
for i, v in enumerate(sorted_r2):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center')

plt.tight_layout()
plt.show()

# Advanced band analysis with permutation importance
print("\n" + "=" * 50)
print("ADVANCED BAND IMPORTANCE ANALYSIS")
print("=" * 50)

# Train a model on all features for permutation analysis
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(rf_model, X_test_scaled, y_test,
                                         n_repeats=10, random_state=42)

# Get the importance scores
importance_scores = perm_importance.importances_mean
importance_std = perm_importance.importances_std

# Create a DataFrame with feature names and importance scores
importance_df = pd.DataFrame({
    'Feature': X_train_columns,
    'Importance': importance_scores,
    'StdDev': importance_std
})

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Group into raw bands and derived indices
raw_bands_importance = importance_df[importance_df['Feature'].isin(raw_bands)]
indices_importance = importance_df[~importance_df['Feature'].isin(raw_bands + ['latitude', 'longitude'])]
location_importance = importance_df[importance_df['Feature'].isin(['latitude', 'longitude'])]

# Print top features
print("\nTop 10 most important features for Nitrogen prediction:")
print(importance_df.head(10))

# Plot permutation importance
plt.figure(figsize=(12, 10))
sorted_idx = importance_scores.argsort()[-20:]  # Top 20 features
plt.barh(range(len(sorted_idx)), importance_scores[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [X_train_columns[i] for i in sorted_idx])
plt.title('Feature Importance for Nitrogen Prediction')
plt.xlabel('Mean Decrease in R² Score')
plt.tight_layout()
plt.show()

# Separate plots for raw bands vs indices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Raw Bands
sorted_bands = raw_bands_importance.sort_values('Importance', ascending=False)
ax1.barh(range(len(sorted_bands)), sorted_bands['Importance'], xerr=sorted_bands['StdDev'],
         align='center', alpha=0.7)
ax1.set_yticks(range(len(sorted_bands)))
ax1.set_yticklabels(sorted_bands['Feature'])
ax1.set_title('Raw Band Importance for N Prediction')
ax1.set_xlabel('Mean Decrease in R² Score')
ax1.grid(True, linestyle='--', alpha=0.7)

# Indices (top 15)
sorted_indices = indices_importance.sort_values('Importance', ascending=False).head(15)
ax2.barh(range(len(sorted_indices)), sorted_indices['Importance'], xerr=sorted_indices['StdDev'],
         align='center', alpha=0.7)
ax2.set_yticks(range(len(sorted_indices)))
ax2.set_yticklabels(sorted_indices['Feature'])
ax2.set_title('Vegetation Index Importance for N Prediction')
ax2.set_xlabel('Mean Decrease in R² Score')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 6. N-Sensitivity Analysis with Cross-Validation
print("\n" + "=" * 50)
print("NITROGEN BAND SENSITIVITY ANALYSIS")
print("=" * 50)

# We'll check if specific bands or indices are particularly sensitive to N values
# by analyzing performance across different ranges of N values

# Group N values into categories
if use_transformed_y:
    # Use the original y values for this analysis
    y_original = np.expm1(y) if skewness > 0 else -np.expm1(-y)
else:
    y_original = y

# Define N level categories
n_percentiles = [0, 33, 66, 100]
n_bins = [np.percentile(y_original, p) for p in n_percentiles]
n_labels = ['Low N', 'Medium N', 'High N']

# Assign categories
n_categories = pd.cut(y_original, bins=n_bins, labels=n_labels, include_lowest=True)

# Create a dict to track performance by N category
n_category_performance = {cat: {'bands': {}, 'indices': {}} for cat in n_labels}


# Function to evaluate performance by N category
def evaluate_by_n_category(features, X, y, n_categories, feature_type='band'):
    """Evaluate model performance for different N level categories"""
    results = {}

    for category in n_labels:
        # Get indices for this category
        cat_indices = np.where(n_categories == category)[0]
        if len(cat_indices) < 10:  # Skip if too few samples
            continue

        # Split data for this category
        X_cat = X[cat_indices]
        y_cat = y[cat_indices]

        if len(X_cat) < 20:  # Need enough samples for train/test
            continue

        # Train-test split
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            X_cat, y_cat, test_size=0.3, random_state=42)

        # Train model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_cat, y_train_cat)

        # Evaluate
        y_pred_cat = rf.predict(X_test_cat)
        r2 = r2_score(y_test_cat, y_pred_cat)
        rmse = np.sqrt(mean_squared_error(y_test_cat, y_pred_cat))

        results[category] = {'r2': r2, 'rmse': rmse}

    return results


# Select the top bands and indices based on earlier importance analysis
top_bands = raw_bands_importance.head(5)['Feature'].tolist()
top_indices = indices_importance.head(5)['Feature'].tolist()

print(f"\nAnalyzing N prediction sensitivity with top bands: {top_bands}")
print(f"And top indices: {top_indices}")

# For each N category, evaluate using top bands and indices
for feature_set, feature_type in [(top_bands, 'bands'), (top_indices, 'indices')]:
    for feature in feature_set:
        # Extract feature column indices
        feature_idx = [list(X_train_columns).index(feature)]

        # Get X data for just this feature
        X_feature = X_train_scaled[:, feature_idx]

        # Evaluate by N category
        results = evaluate_by_n_category(
            [feature], X_feature, y_train, n_categories[y_train.index], feature_type)

        # Store results
        for category, metrics in results.items():
            n_category_performance[category][feature_type][feature] = metrics

# Plot the results
plt.figure(figsize=(14, 10))

# For each N category
for i, category in enumerate(n_labels):
    plt.subplot(3, 1, i + 1)

    # Get band data for this category
    if category in n_category_performance:
        band_data = n_category_performance[category]['bands']
        index_data = n_category_performance[category]['indices']

        # Extract band names and R² scores
        band_names = list(band_data.keys())
        band_r2 = [band_data[b]['r2'] for b in band_names]

        # Extract index names and R² scores
        index_names = list(index_data.keys())
        index_r2 = [index_data[i]['r2'] for i in index_names]

        # Plot bands
        x = np.arange(len(band_names))
        plt.bar(x, band_r2, width=0.4, label='Bands', color='skyblue')

        # Plot indices
        if index_names:
            x2 = np.arange(len(band_names), len(band_names) + len(index_names))
            plt.bar(x2, index_r2, width=0.4, label='Indices', color='coral')
            plt.xticks(np.concatenate([x, x2]), band_names + index_names, rotation=45)
        else:
            plt.xticks(x, band_names, rotation=45)

        plt.title(f'R² Scores for {category} Prediction')
        plt.ylabel('R² Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    else:
        plt.text(0.5, 0.5, f'Insufficient data for {category}',
                 horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.show()

# Define models to try
models = {
    'XGBoost (Tuned)': XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    ),
    'LightGBM (Tuned)': lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    ),
    'Random Forest (Tuned)': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    ),
    'Gradient Boosting (Tuned)': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
}

# Evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    results[name] = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, name)

# Find the best model based on test R²
best_model_name = max(results, key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with Test R² = {results[best_model_name]['test_r2']:.4f}")

# Cross-validation of the best model
cv_scores = cross_val_score(
    best_model,
    scaler.transform(X_selected),
    y_transformed if use_transformed_y else y,
    cv=5,
    scoring='r2'
)
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Create an Ensemble Model (Stacking)
base_models = [
    ('xgb', models['XGBoost (Tuned)']),
    ('lgbm', models['LightGBM (Tuned)']),
    ('rf', models['Random Forest (Tuned)'])
]
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(),
    cv=5
)

# Evaluate stacking model
stacking_results = evaluate_model(
    stacking_model,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    "Stacking Ensemble"
)

# Add to results
results["Stacking Ensemble"] = stacking_results

# Update best model if stacking is better
if stacking_results['test_r2'] > results[best_model_name]['test_r2']:
    best_model_name = "Stacking Ensemble"
    best_model = stacking_model
    print(f"Stacking Ensemble is now the best model!")

# Visualize Results
# Compare model performances
model_names = list(results.keys())
test_r2_scores = [results[name]['test_r2'] for name in model_names]
test_rmse_scores = [results[name]['test_rmse'] for name in model_names]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bars = plt.bar(model_names, test_r2_scores)
plt.title('Test R² Score by Model')
plt.xticks(rotation=45, ha='right')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{test_r2_scores[i]:.3f}", ha='center')

plt.subplot(1, 2, 2)
bars = plt.bar(model_names, test_rmse_scores)
plt.title('Test RMSE by Model')
plt.xticks(rotation=45, ha='right')
plt.ylabel('RMSE')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{test_rmse_scores[i]:.3f}", ha='center')

plt.tight_layout()
plt.show()

# Plot actual vs predicted values for the best model
best_y_test = results[best_model_name]['y_test_actual']
best_y_pred = results[best_model_name]['y_test_pred']

plt.figure(figsize=(10, 8))
plt.scatter(best_y_test, best_y_pred, alpha=0.7)
plt.plot([best_y_test.min(), best_y_test.max()], [best_y_test.min(), best_y_test.max()], 'r--')

# Add the ±20% lines
y_min, y_max = best_y_test.min(), best_y_test.max()
shift = 0.2 * (y_max - y_min)
plt.plot([y_min, y_max], [y_min + shift, y_max + shift], 'k--', alpha=0.5)
plt.plot([y_min, y_max], [y_min - shift, y_max - shift], 'k--', alpha=0.5)

plt.xlabel('Actual Nitrogen (N)')
plt.ylabel('Predicted Nitrogen (N)')
plt.title(f'Actual vs. Predicted N Values - {best_model_name}\nTest R² = {results[best_model_name]["test_r2"]:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate percentage of points within ±20% bounds
upper_bound = best_y_test + shift
lower_bound = best_y_test - shift
within_bounds = np.sum((best_y_pred <= upper_bound) & (best_y_pred >= lower_bound))
percentage_within = (within_bounds / len(best_y_test)) * 100
print(f"\nPercentage of predictions within ±20% bounds: {percentage_within:.2f}%")

# Feature Importance for the best model (if supported)
if hasattr(best_model, 'feature_importances_') or (
        hasattr(best_model, 'estimators_') and hasattr(best_model.estimators_[0], 'feature_importances_')):
    # For direct models
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    # For stacked models
    elif best_model_name == "Stacking Ensemble":
        # Get feature importances from the first base model (XGBoost)
        importances = best_model.estimators_[0][1].feature_importances_

    # Sort feature importance
    indices = np.argsort(importances)[::-1]
    feature_names = X_selected.columns

    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance - {best_model_name}')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

# Save the best model
import joblib

joblib.dump(best_model, 'best_nitrogen_prediction_model.pkl')
joblib.dump(scaler, 'nitrogen_feature_scaler.pkl')
joblib.dump(selected_features, 'nitrogen_selected_features.pkl')

print("\nModel saved as 'best_nitrogen_prediction_model.pkl'")
print("Scaler saved as 'nitrogen_feature_scaler.pkl'")
print("Selected features saved as 'nitrogen_selected_features.pkl'")


# Function for making predictions on new data
def predict_nitrogen(new_data, model=best_model, scaler=scaler, selected_features=selected_features):
    """
    Make nitrogen predictions on new data

    Parameters:
    new_data (DataFrame): DataFrame containing all the required bands and indices
    model: Trained model
    scaler: Fitted scaler
    selected_features: List of selected features

    Returns:
    array: Predicted nitrogen values
    """
    # Select features
    X_new = new_data[selected_features]

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    predictions = model.predict(X_new_scaled)

    # Transform back if log transformation was applied
    if use_transformed_y:
        if skewness > 0:
            predictions = np.expm1(predictions)
        else:
            predictions = -np.expm1(-predictions)

    return predictions


print("\nUse the predict_nitrogen() function to make predictions on new data")
