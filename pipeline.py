"""
EEG Mental Workload Classification Pipeline
Based on the STEW (Simultaneous Task EEG Workload) Dataset
INEL 4998 - Undergraduate Research | UPRM - AIIG Lab
PI: Favian O. Díaz | Advisor: Prof. Vidya Manian

DATASET: STEW - IEEE DataPort (open access)
https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset

HOW TO DOWNLOAD:
1. Go to the URL above and create a free IEEE account
2. Download the ZIP and extract into a folder called "STEW_data/"
3. The folder should contain files like: sub01_lo.txt, sub01_hi.txt, sub02_lo.txt, etc.
4. Also download "ratings.csv" (the workload self-ratings)

The pipeline will run in DEMO MODE with synthetic data if STEW_data/ is not found.
Demo mode mimics the real data structure so you can test everything before downloading.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = "."          # Folder with extracted STEW files
FS = 128                         # Sampling frequency (Hz) — STEW uses 128 Hz
N_CHANNELS = 14                  # Emotiv EPOC channels
CHANNEL_NAMES = ['AF3','F7','F3','FC5','T7','P7','O1',
                 'O2','P8','T8','FC6','F4','F8','AF4']
WINDOW_SEC = 2                   # Seconds per analysis window
WINDOW_SAMPLES = WINDOW_SEC * FS # Samples per window (256)
OVERLAP = 0.5                    # 50% overlap between windows

# EEG frequency bands (Hz)
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════

def load_stew_dataset(data_dir):
    """
    Load the STEW dataset from the extracted folder.
    Returns a list of (eeg_array, label) tuples.
    
    label = 0 → REST (low workload)
    label = 1 → TASK (high workload, SIMKAP multitasking test)
    """
    subjects = []
    
    if not os.path.exists(data_dir):
        print(f"[INFO] '{data_dir}/' not found. Running in DEMO MODE with synthetic data.")
        print("[INFO] See docstring at top of file for download instructions.\n")
        return generate_demo_data()
    
    # Find subject files
    files = os.listdir(data_dir)
    subject_ids = sorted(set(f.split('_')[0] for f in files if f.endswith('.txt')))
    
    for sub_id in subject_ids:
        lo_file = os.path.join(data_dir, f"{sub_id}_lo.txt")  # rest
        hi_file = os.path.join(data_dir, f"{sub_id}_hi.txt")  # task
        
        if not (os.path.exists(lo_file) and os.path.exists(hi_file)):
            continue
        
        try:
            lo_data = np.loadtxt(lo_file)   # shape: (samples, 14)
            hi_data = np.loadtxt(hi_file)
            
            # Truncate to 2.5 min × 128 Hz = 19200 samples
            lo_data = lo_data[:19200, :N_CHANNELS]
            hi_data = hi_data[:19200, :N_CHANNELS]
            
            subjects.append({'id': sub_id, 'rest': lo_data, 'task': hi_data})
            
        except Exception as e:
            print(f"  [WARNING] Could not load {sub_id}: {e}")
    
    print(f"[INFO] Loaded {len(subjects)} subjects from {data_dir}/")
    return subjects


def generate_demo_data(n_subjects=20):
    """
    Generate synthetic EEG-like data that mimics STEW structure.
    REST: higher alpha power | TASK: higher theta + beta power (known workload markers)
    """
    print("[DEMO] Generating synthetic EEG data (20 subjects, 2.5 min each condition)...")
    subjects = []
    n_samples = 19200  # 2.5 min × 128 Hz
    t = np.linspace(0, n_samples / FS, n_samples)
    
    for i in range(n_subjects):
        sub_data = {}
        for condition, params in [('rest',  {'alpha': 2.0, 'theta': 0.5, 'beta': 0.5}),
                                   ('task',  {'alpha': 0.5, 'theta': 1.5, 'beta': 1.5})]:
            eeg = np.zeros((n_samples, N_CHANNELS))
            for ch in range(N_CHANNELS):
                # Simulate alpha oscillation (8-12 Hz)
                sig = (params['alpha'] * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi))
                # Theta (4-7 Hz)
                sig += (params['theta'] * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi))
                # Beta (13-25 Hz)
                sig += (params['beta'] * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi))
                # Pink noise
                sig += 0.3 * np.cumsum(np.random.randn(n_samples)) / np.sqrt(n_samples)
                # Subject-level variability
                sig += np.random.randn() * 0.5
                eeg[:, ch] = sig
            sub_data[condition] = eeg
        
        subjects.append({'id': f'sub{i+1:02d}', 'rest': sub_data['rest'], 'task': sub_data['task']})
    
    return subjects


# ═══════════════════════════════════════════════════════
# SECTION 2: PREPROCESSING
# ═══════════════════════════════════════════════════════

def bandpass_filter(data, low, high, fs=FS, order=4):
    """Apply Butterworth bandpass filter to EEG data."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return sp_signal.filtfilt(b, a, data, axis=0)


def remove_dc_offset(data):
    """Remove DC offset (mean subtraction per channel)."""
    return data - data.mean(axis=0)


def preprocess_eeg(eeg, fs=FS):
    """
    Basic EEG preprocessing pipeline:
    1. Remove DC offset
    2. Bandpass filter (1–45 Hz) to remove drift and line noise
    3. Simple amplitude thresholding for artifact rejection (±100 µV)
    """
    eeg = remove_dc_offset(eeg)
    eeg = bandpass_filter(eeg, 1.0, 45.0, fs)
    
    # Amplitude-based artifact rejection: clip extreme values
    threshold = 100.0  # µV
    artifact_mask = np.any(np.abs(eeg) > threshold, axis=1)
    if artifact_mask.sum() > 0:
        # Interpolate artifacted samples with neighboring clean samples
        clean_indices = np.where(~artifact_mask)[0]
        for ch in range(eeg.shape[1]):
            eeg[artifact_mask, ch] = np.interp(
                np.where(artifact_mask)[0], clean_indices, eeg[clean_indices, ch]
            )
    
    return eeg


# ═══════════════════════════════════════════════════════
# SECTION 3: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def compute_band_power(epoch, fs=FS):
    """
    Compute spectral power in each EEG band using Welch's method.
    Returns a dict: {band_name: power_per_channel}
    
    Welch's method averages multiple shorter FFTs to reduce noise —
    more reliable than a single FFT for short epochs.
    """
    n_samples, n_ch = epoch.shape
    nperseg = min(n_samples, 128)  # 1-second segments at 128 Hz
    
    freqs, psd = sp_signal.welch(epoch, fs=fs, nperseg=nperseg, axis=0)
    
    band_powers = {}
    for band_name, (low, high) in BANDS.items():
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) == 0:
            band_powers[band_name] = np.zeros(n_ch)
        else:
            # Average power across frequency bin (trapezoidal integration)
            band_powers[band_name] = np.trapezoid(psd[idx, :], freqs[idx], axis=0)
    
    return band_powers, freqs, psd


def compute_features_from_epoch(epoch, fs=FS):
    """
    Extract all features from a single EEG epoch (window).
    
    Features computed:
    1. Absolute band power (theta, alpha, beta) per channel → 42 features
    2. Relative band power (band / total) → 42 features
    3. Frontal theta / parietal alpha ratio → key workload marker from proposal
    4. Spectral entropy per channel → 14 features
    5. Statistical features (variance, skewness) → 28 features
    
    Total: ~131 features per window
    """
    features = []
    feature_names = []
    
    band_powers, freqs, psd = compute_band_power(epoch, fs)
    
    # Total power for relative computation
    total_power = sum(band_powers[b] for b in BANDS)
    total_power = np.where(total_power == 0, 1e-10, total_power)
    
    # ── 1. Absolute band powers (theta, alpha, beta) ──
    for band in ['theta', 'alpha', 'beta']:
        for ch, ch_name in enumerate(CHANNEL_NAMES):
            features.append(np.log1p(band_powers[band][ch]))  # log for normalization
            feature_names.append(f'abs_{band}_{ch_name}')
    
    # ── 2. Relative band powers ──
    for band in ['theta', 'alpha', 'beta']:
        rel = band_powers[band] / total_power
        for ch, ch_name in enumerate(CHANNEL_NAMES):
            features.append(rel[ch])
            feature_names.append(f'rel_{band}_{ch_name}')
    
    # ── 3. Frontal theta / parietal alpha ratio (PCLI proxy) ──
    # From proposal Section 3.5: "frontal theta/parietal alpha ratio (established workload marker)"
    frontal_channels = [CHANNEL_NAMES.index(c) for c in ['AF3', 'F3', 'F4', 'AF4', 'FC5', 'FC6'] 
                        if c in CHANNEL_NAMES]
    parietal_channels = [CHANNEL_NAMES.index(c) for c in ['P7', 'P8'] 
                         if c in CHANNEL_NAMES]
    
    if frontal_channels and parietal_channels:
        frontal_theta = band_powers['theta'][frontal_channels].mean()
        parietal_alpha = band_powers['alpha'][parietal_channels].mean()
        ratio = frontal_theta / (parietal_alpha + 1e-10)
        features.append(np.log1p(ratio))
        feature_names.append('frontal_theta_parietal_alpha_ratio')
    
    # ── 4. Spectral entropy ──
    for ch, ch_name in enumerate(CHANNEL_NAMES):
        psd_norm = psd[:, ch] / (psd[:, ch].sum() + 1e-10)
        spec_ent = entropy(psd_norm + 1e-10)
        features.append(spec_ent)
        feature_names.append(f'spectral_entropy_{ch_name}')
    
    # ── 5. Temporal statistics ──
    for ch, ch_name in enumerate(CHANNEL_NAMES):
        features.append(np.var(epoch[:, ch]))
        feature_names.append(f'variance_{ch_name}')
    
    from scipy.stats import skew
    for ch, ch_name in enumerate(CHANNEL_NAMES):
        features.append(skew(epoch[:, ch]))
        feature_names.append(f'skewness_{ch_name}')
    
    return np.array(features), feature_names


def extract_windows(eeg, label, window_samples=WINDOW_SAMPLES, overlap=OVERLAP):
    """
    Segment EEG into overlapping windows and extract features from each.
    Returns feature matrix X and label vector y.
    """
    step = int(window_samples * (1 - overlap))
    X_windows, y_windows = [], []
    
    start = 0
    while start + window_samples <= eeg.shape[0]:
        epoch = eeg[start:start + window_samples, :]
        feats, _ = compute_features_from_epoch(epoch)
        X_windows.append(feats)
        y_windows.append(label)
        start += step
    
    return np.array(X_windows), np.array(y_windows)


def build_feature_matrix(subjects):
    """
    Build full feature matrix from all subjects.
    Returns X (features), y (labels), groups (subject IDs for LOSO CV).
    """
    print("\n[INFO] Extracting features from all subjects...")
    all_X, all_y, all_groups = [], [], []
    feature_names = None
    
    for subj in subjects:
        sub_id = subj['id']
        
        # Preprocess
        rest_eeg = preprocess_eeg(subj['rest'])
        task_eeg = preprocess_eeg(subj['task'])
        
        # Extract windows: rest=0, task=1
        X_rest, y_rest = extract_windows(rest_eeg, label=0)
        X_task, y_task = extract_windows(task_eeg, label=1)
        
        X_sub = np.vstack([X_rest, X_task])
        y_sub = np.concatenate([y_rest, y_task])
        
        # Get feature names from first subject
        if feature_names is None:
            _, feature_names = compute_features_from_epoch(rest_eeg[:WINDOW_SAMPLES, :])
        
        all_X.append(X_sub)
        all_y.append(y_sub)
        all_groups.extend([sub_id] * len(y_sub))
        
        print(f"  {sub_id}: {len(X_rest)} rest windows + {len(X_task)} task windows = {len(y_sub)} total")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)
    
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n[INFO] Total dataset: {X.shape[0]} windows × {X.shape[1]} features")
    print(f"[INFO] Class balance: {(y==0).sum()} rest ({100*(y==0).mean():.1f}%) | "
          f"{(y==1).sum()} task ({100*(y==1).mean():.1f}%)")
    
    return X, y, groups, feature_names


# ═══════════════════════════════════════════════════════
# SECTION 4: MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════

def build_models():
    """
    Define the three classifiers from the proposal (Section 3.5):
    - Support Vector Machine (SVM) with RBF kernel
    - Random Forest
    - Logistic Regression
    
    Each wrapped in a Pipeline with StandardScaler.
    """
    models = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=SEED, n_jobs=-1))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
        ])
    }
    return models


def evaluate_models(X, y, models, cv_folds=5):
    """
    Evaluate all models using Stratified K-Fold cross-validation.
    Reports accuracy, F1-score, and ROC-AUC for each model.
    """
    print(f"\n[INFO] Running {cv_folds}-fold Stratified Cross-Validation...\n")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    results = {}
    
    for name, model in models.items():
        print(f"  Training: {name}...")
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        results[name] = {
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc':  auc,
            'y_pred':   y_pred,
            'y_proba':  y_proba
        }
        print(f"    Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    
    return results


def get_feature_importance(X, y, feature_names):
    """
    Compute feature importance using Random Forest.
    Returns sorted DataFrame of top features.
    """
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


# ═══════════════════════════════════════════════════════
# SECTION 5: VISUALIZATION
# ═══════════════════════════════════════════════════════

def plot_all_results(results, importance_df, subjects, X, y):
    """Generate a comprehensive results figure with 4 subplots."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('EEG Mental Workload Classification — STEW Dataset\nINEL 4998 | UPRM AIIG Lab',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    colors = {'SVM (RBF)': '#2196F3', 'Random Forest': '#4CAF50', 'Logistic Regression': '#FF9800'}
    
    # ── Plot 1: Model Comparison Bar Chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        bars = ax1.bar(x + i*width, vals, width, label=name,
                       color=colors[name], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.grid(axis='y', alpha=0.3)
    
    # ── Plot 2: Confusion Matrix (best model) ──
    ax2 = fig.add_subplot(gs[0, 1])
    best_model = max(results, key=lambda k: results[k]['roc_auc'])
    cm = confusion_matrix(y, results[best_model]['y_pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax2,
                cmap='Blues', linewidths=0.5,
                xticklabels=['REST\n(low workload)', 'TASK\n(high workload)'],
                yticklabels=['REST\n(low workload)', 'TASK\n(high workload)'])
    ax2.set_title(f'Confusion Matrix — {best_model}\n(normalized, best model by ROC-AUC)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # ── Plot 3: Top 20 Feature Importances ──
    ax3 = fig.add_subplot(gs[1, 0])
    top_n = 20
    top_feats = importance_df.head(top_n)
    
    # Color by band type
    feat_colors = []
    for fname in top_feats['feature']:
        if 'theta' in fname:    feat_colors.append('#E91E63')
        elif 'alpha' in fname:  feat_colors.append('#9C27B0')
        elif 'beta'  in fname:  feat_colors.append('#3F51B5')
        elif 'ratio' in fname:  feat_colors.append('#FF5722')
        else:                   feat_colors.append('#607D8B')
    
    bars = ax3.barh(range(top_n), top_feats['importance'], color=feat_colors, alpha=0.85)
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(top_feats['feature'], fontsize=7)
    ax3.invert_yaxis()
    ax3.set_xlabel('Feature Importance (Random Forest)')
    ax3.set_title(f'Top {top_n} Most Informative Features')
    ax3.grid(axis='x', alpha=0.3)
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E91E63', label='Theta'),
                       Patch(facecolor='#9C27B0', label='Alpha'),
                       Patch(facecolor='#3F51B5', label='Beta'),
                       Patch(facecolor='#FF5722', label='Theta/Alpha Ratio'),
                       Patch(facecolor='#607D8B', label='Other')]
    ax3.legend(handles=legend_elements, fontsize=7, loc='lower right')
    
    # ── Plot 4: Mean Band Power by Condition ──
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Compute mean power for each condition from a sample of windows
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Use first subject as example
    subj = subjects[0]
    rest_proc = preprocess_eeg(subj['rest'])
    task_proc = preprocess_eeg(subj['task'])
    
    rest_epoch = rest_proc[:WINDOW_SAMPLES, :]
    task_epoch = task_proc[:WINDOW_SAMPLES, :]
    
    rest_powers, _, _ = compute_band_power(rest_epoch)
    task_powers, _, _ = compute_band_power(task_epoch)
    
    x_bands = np.arange(len(band_names))
    width = 0.35
    
    rest_means = [np.log1p(rest_powers[b].mean()) for b in band_names]
    task_means = [np.log1p(task_powers[b].mean()) for b in band_names]
    
    ax4.bar(x_bands - width/2, rest_means, width, label='REST (low workload)',
            color='#42A5F5', alpha=0.85, edgecolor='white')
    ax4.bar(x_bands + width/2, task_means, width, label='TASK (high workload)',
            color='#EF5350', alpha=0.85, edgecolor='white')
    
    ax4.set_xlabel('EEG Frequency Band')
    ax4.set_ylabel('Log Power (µV²/Hz)')
    ax4.set_title('Mean EEG Band Power: REST vs TASK\n(example subject, averaged across channels)')
    ax4.set_xticks(x_bands)
    ax4.set_xticklabels([b.capitalize() for b in band_names])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_figure.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[INFO] Figure saved → {OUTPUT_DIR}/results_figure.png")


# ═══════════════════════════════════════════════════════
# SECTION 6: RESULTS SUMMARY
# ═══════════════════════════════════════════════════════

def print_summary(results, importance_df):
    """Print a clean summary table of results."""
    print("\n" + "="*65)
    print("  RESULTS SUMMARY — EEG Mental Workload Classification")
    print("="*65)
    print(f"\n{'Model':<22} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
    print("-"*55)
    
    best_model = max(results, key=lambda k: results[k]['roc_auc'])
    
    for name, res in results.items():
        marker = " ← best" if name == best_model else ""
        print(f"{name:<22} {res['accuracy']:>10.3f} {res['f1_score']:>10.3f} {res['roc_auc']:>10.3f}{marker}")
    
    print("\n" + "─"*55)
    print("Top 5 most informative features (Random Forest):")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:<40} {row['importance']:.4f}")
    
    print("\nConnection to HLR (High-Leverage Readiness) System:")
    print("  • Frontal theta/parietal alpha ratio = proxy for PCLI")
    print("  • High beta power = elevated cognitive arousal (task demand)")
    print("  • Alpha suppression = active cognitive processing")
    print("  • These features will be extracted from pitcher EEG during")
    print("    bullpen sessions (Phase 2, next semester)")
    print("="*65)
    
    # Save results to CSV
    summary = []
    for name, res in results.items():
        summary.append({'Model': name, 'Accuracy': res['accuracy'],
                        'F1_Score': res['f1_score'], 'ROC_AUC': res['roc_auc']})
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print(f"\n[INFO] Results saved → {OUTPUT_DIR}/model_results.csv")
    print(f"[INFO] Feature importance → {OUTPUT_DIR}/feature_importance.csv")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  EEG Mental Workload ML Pipeline — INEL 4998 / AIIG Lab  ║")
    print("║  UPRM | Advisor: Prof. Vidya Manian                      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    # Step 1: Load data
    subjects = load_stew_dataset(DATA_DIR)
    
    # Step 2: Build feature matrix
    X, y, groups, feature_names = build_feature_matrix(subjects)
    
    # Step 3: Evaluate models
    models = build_models()
    results = evaluate_models(X, y, models)
    
    # Step 4: Feature importance
    print("\n[INFO] Computing feature importance...")
    importance_df = get_feature_importance(X, y, feature_names)
    
    # Step 5: Visualize
    print("[INFO] Generating figures...")
    plot_all_results(results, importance_df, subjects, X, y)
    
    # Step 6: Summary
    print_summary(results, importance_df)
    
    print("\n✓ Pipeline complete. Check the 'results/' folder for outputs.")
    return results, importance_df


if __name__ == "__main__":
    results, importance_df = main()