import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

config = {
    "font.family": 'Arial',
    "font.weight": 'normal',
    "font.size": 14,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "axes.unicode_minus": False,
    "savefig.dpi": 600
}
rcParams.update(config)

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'test_path': r'C:\Users\86136\Desktop\ML-ZK\test.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

best_params_svr = {
    'kernel': 'rbf',
    'C': 9974.37312046307,
    'gamma': 0.0048580560622156505,
    'epsilon': 0.013751484997828403
}

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')

    train_df = train_df[train_df[DATA_CONFIG['target']] > -500]
    test_df = test_df[test_df[DATA_CONFIG['target']] > -500]

    features = [c for c in train_df.columns if c not in [DATA_CONFIG['target'], DATA_CONFIG['exclude']]]
    return (train_df[features], train_df[DATA_CONFIG['target']],
            test_df[features], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = SVR(**best_params_svr)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_preds = model.predict(X_train)

def calc_prediction_intervals_scaled(y_train, y_train_pred, y_test_pred, scale_factor=1.2):
    residuals = y_train - y_train_pred
    sigma_residual = np.std(residuals, ddof=1) * scale_factor
    z_score = 1.645

    test_lower = y_test_pred - z_score * sigma_residual
    test_upper = y_test_pred + z_score * sigma_residual
    return test_lower, test_upper

lower_bound, upper_bound = calc_prediction_intervals_scaled(
    y_train, train_preds, y_pred, scale_factor=1.2
)

sample_idx = np.arange(len(y_test))

fig, ax = plt.subplots(figsize=(14, 9))

color_pi = '#87CEFA'
color_obs = '#d73027'
color_pred = '#084594'

ax.fill_between(sample_idx, lower_bound, upper_bound,
                color=color_pi, alpha=0.4, zorder=1, label='90% Prediction Interval')

ax.plot(sample_idx, lower_bound, color=color_pi, lw=2.0, linestyle='--', alpha=0.8, zorder=1)
ax.plot(sample_idx, upper_bound, color=color_pi, lw=2.0, linestyle='--', alpha=0.8, zorder=1)

ax.plot(sample_idx, y_pred, color=color_pred, linewidth=2.0, alpha=0.6, zorder=2)
ax.scatter(sample_idx, y_pred, color=color_pred, s=80, marker='o',
           edgecolors='white', linewidths=0.5, zorder=4, label='Predicted Value')

ax.scatter(sample_idx, y_test, facecolors='none', edgecolors=color_obs,
           s=180, marker='*', linewidths=1.5, zorder=3, label='Observed Value')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_xlabel('Predicted sample', fontsize=24, fontweight='bold', labelpad=12)
ax.set_ylabel('Predicted ΔAs (%)', fontsize=24, fontweight='bold', labelpad=12)

y_all_min = min(lower_bound.min(), y_test.min())
y_all_max = max(upper_bound.max(), y_test.max())
margin = (y_all_max - y_all_min) * 0.15
ax.set_ylim(y_all_min - margin, y_all_max + margin)

ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

ax.legend(loc='lower left', frameon=True, fontsize=24,
          facecolor='white', framealpha=0.9,
          prop={'family': 'Arial', 'weight': 'bold', 'size': 24})

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties({'family': 'Arial', 'size': 20, 'weight': 'normal'})

ax.tick_params(axis='both', which='major', length=7, width=1.5, pad=8)

ax.grid(False)
plt.tight_layout()

save_path = r'C:\Users\86136\Desktop\ML-ZK\SVR-As-Prediction interval.tiff'
plt.savefig(save_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()