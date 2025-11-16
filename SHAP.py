#pip install scikit-learn==1.1.3
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from xgboost import XGBModel
from lightgbm import LGBMModel
try:
    from catboost import CatBoost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

csv_path = '/home/featurize/rad_featuresALLSHAP.csv'
model_path = '/home/featurize/models/ClinicRandomForest_label.pkl'

df = pd.read_csv(csv_path)
group_col = 'group'
target_col = df.columns[-1]   # 或直接写字符串
exclude_cols = ['ID', group_col, target_col] if 'ID' in df.columns else [group_col, target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]
print('最终用于分析的特征:', feature_cols)

train_df = df[df[group_col] == 'train'].copy()
test_df = df[df[group_col] == 'test'].copy()
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = joblib.load(model_path)

def is_tree_model(m):
    if isinstance(m, (XGBModel, LGBMModel,
                      RandomForestClassifier, RandomForestRegressor,
                      GradientBoostingClassifier, GradientBoostingRegressor,)):
        return True
    if CATBOOST_AVAILABLE and isinstance(m, CatBoost):
        return True
    return False

tree_model_flag = is_tree_model(model)
print('模型属于树模型:', tree_model_flag)

background_data = shap.utils.sample(X_train, min(100, len(X_train)), random_state=42)

if tree_model_flag:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = explainer.expected_value
else:
    def model_predict(data):
        return model.predict(data)
    explainer = shap.KernelExplainer(model_predict, background_data)
    shap_values = explainer.shap_values(X_test, nsamples=100)
    expected_value = explainer.expected_value

print('shap_values shape:', np.array(shap_values).shape)
print('X_test feature count:', X_test.shape[1])

# ------------------ 关键兼容代码 ------------------
# 若输出为 (n_samples, n_features, n_classes)，取正类（通常类别1）的shap值
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == 2:
    shap_values_for_plot = shap_values[:, :, 1]
    expected_value_for_plot = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
else:
    shap_values_for_plot = shap_values
    expected_value_for_plot = expected_value
print('shap_values_for_plot.shape:', np.array(shap_values_for_plot).shape)
print('expected_value_for_plot:', expected_value_for_plot)
# -------------------------------------------------

plt.rcParams.update({
    'font.size': 10,
    'axes.titlepad': 12,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': True,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# ================== Beeswarm Plot ==================
plt.figure(figsize=(10, 6), dpi=600)
shap.summary_plot(
    shap_values_for_plot, X_test,
    plot_type='dot', max_display=10, show=False
)
ax = plt.gca()
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}" if x >= 0 else f"-{abs(x):.3f}"))
plt.subplots_adjust(left=0.3, right=0.85, top=0.95, bottom=0.1)
plt.savefig('/home/featurize/output/beeswarm_plot.png', bbox_inches='tight', dpi=600, facecolor='white')
plt.close()

# ================== Bar Plot ==================
plt.figure(figsize=(12, 8), dpi=600)
shap.summary_plot(shap_values_for_plot, X_test, plot_type='bar', max_display=10, show=False)
ax = plt.gca()
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=12)
ax.set_xlabel("mean(|SHAP value|)", fontsize=12, labelpad=10)
ax.set_ylabel("Features", fontsize=12, labelpad=10)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
plt.subplots_adjust(left=0.4, right=0.85, top=0.95, bottom=0.1)
plt.savefig('/home/featurize/output/bar_plot.png', bbox_inches='tight', dpi=600, facecolor='white')
plt.close()

# ================== Waterfall Plot ==================
sample_idx = 0
explanation = shap.Explanation(
    values=shap_values_for_plot[sample_idx],
    base_values=expected_value_for_plot,
    data=X_test.iloc[sample_idx],
    feature_names=list(X_test.columns)
)
plt.figure(figsize=(8, 6), dpi=600)
shap.plots.waterfall(explanation, max_display=10, show=False)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}" if x >= 0 else f"-{abs(x):.3f}"))
ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(8)
    label.set_weight('normal')
plt.savefig('/home/featurize/output/waterfall_plot.png', bbox_inches='tight', dpi=600, facecolor='white')
plt.close()

# ================== Force Plot ==================
try:
    force_plot = shap.force_plot(
        base_value=expected_value_for_plot,
        shap_values=shap_values_for_plot[sample_idx],
        features=X_test.iloc[sample_idx],
        feature_names=X_test.columns,
        matplotlib=False
    )
    shap.save_html("/home/featurize/output/force_plot.html", force_plot)
except Exception as e:
    print("Force plot generation exception:", e)