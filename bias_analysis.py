import gc
import os
import numpy as np
import pandas as pd

try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


# ── Dataset konfiqurasiyalari ────────────────────────────────────────────────

CONFIGS = {
    'adult': {
        'protected':        'sex',
        'label':            'income',
        'privileged_val':   1.0,
        'unprivileged_val': 0.0,
        'favorable_label':  1.0,
        'categorical': [
            'workclass', 'education', 'marital-status',
            'occupation', 'relationship', 'race', 'native-country'
        ],
        'encode': {
            'sex':    {'Male': 1.0, 'Female': 0.0},
            'income': {'>50K': 1.0, '<=50K': 0.0,
                       '>50K.': 1.0, '<=50K.': 0.0},
        }
    },
    'compas': {
        'protected':        'race',
        'label':            'yeniden_cinayat',
        'privileged_val':   1.0,
        'unprivileged_val': 0.0,
        'favorable_label':  0.0,
        'categorical': [
            'ittiham_derece', 'risk_qiymeti', 'sex'
        ],
        'encode': {
            'race': {'Caucasian': 1.0, 'African-American': 0.0},
        }
    }
}


def _detect_config(df: pd.DataFrame) -> dict:
    cols = set(df.columns.str.strip())
    if 'income' in cols:
        return CONFIGS['adult']
    return CONFIGS['compas']


def _csv_to_aif360(df: pd.DataFrame, cfg: dict) -> BinaryLabelDataset:
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df.columns = df.columns.str.strip()

    for col, mapping in cfg['encode'].items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    drop_cols = [c for c in [cfg['protected'], cfg['label']] if c in df.columns]
    df = df.dropna(subset=drop_cols)

    cat_cols = [c for c in cfg['categorical'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=np.float32)

    arr = df.to_numpy(dtype=np.float32)
    df  = pd.DataFrame(arr, columns=list(df.columns))

    return BinaryLabelDataset(
        df=df,
        label_names=[cfg['label']],
        protected_attribute_names=[cfg['protected']],
        favorable_label=cfg['favorable_label'],
        unfavorable_label=1.0 - cfg['favorable_label']
    )


def _align(ds1: BinaryLabelDataset, ds2: BinaryLabelDataset):
    common = sorted(set(ds1.feature_names) & set(ds2.feature_names))

    def _filter(ds, keep):
        idx = [ds.feature_names.index(c) for c in keep]
        out = ds.copy()
        out.features      = ds.features[:, idx]
        out.feature_names = keep
        return out

    return _filter(ds1, common), _filter(ds2, common)


def _fit_predict(train_ds, test_ds, sample_weight=None):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_ds.features)
    X_test  = scaler.transform(test_ds.features)
    y_train = train_ds.labels.ravel()
    y_test  = test_ds.labels.ravel()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)

    pred_ds        = test_ds.copy()
    pred_ds.labels = y_pred.reshape(-1, 1)
    return accuracy_score(y_test, y_pred), pred_ds, model, scaler


def _bias_metrics(test_ds, pred_ds, cfg):
    priv   = [{cfg['protected']: cfg['privileged_val']}]
    unpriv = [{cfg['protected']: cfg['unprivileged_val']}]

    dm = BinaryLabelDatasetMetric(pred_ds,
                                  unprivileged_groups=unpriv,
                                  privileged_groups=priv)
    cm = ClassificationMetric(test_ds, pred_ds,
                               unprivileged_groups=unpriv,
                               privileged_groups=priv)

    pa = pred_ds.protected_attributes[:, 0]

    try:
        pp_priv   = cm.positive_predictive_value(privileged=True)
        pp_unpriv = cm.positive_predictive_value(privileged=False)
        pred_parity = float(pp_priv - pp_unpriv)
    except Exception:
        pred_parity = float('nan')

    return {
        'disparate_impact':   dm.disparate_impact(),
        'statistical_parity': dm.mean_difference(),
        'equal_opportunity':  cm.equal_opportunity_difference(),
        'avg_odds_diff':      cm.average_odds_difference(),
        'predictive_parity':  pred_parity,
        'privileged_rate':   float(pred_ds.labels[pa == cfg['privileged_val']].mean()),
        'unprivileged_rate': float(pred_ds.labels[pa == cfg['unprivileged_val']].mean()),
    }


def _shorten(name):
    name = name.replace('marital-status_', 'marital:')
    name = name.replace('native-country_', 'country:')
    name = name.replace('ittiham_derece_', 'ittiham:')
    name = name.replace('risk_qiymeti_', 'risk:')
    name = name.replace('relationship_', 'rel:')
    name = name.replace('occupation_', 'iş:')
    name = name.replace('education_', 'təh:')
    name = name.replace('workclass_', 'sektor:')
    return name[:32]


def _compute_shap(model, scaler, train_ds, test_ds, top_n=12):
    """
    SHAP LinearExplainer ile xususiyyetlerin modelə tesirini hesablayir.
    SHAP movcud deyilse ve ya xeta olsa, logistic regression koefficientleri
    istifade edilir (linear modeller ucun riyazi olaraq ekvivalentdir).
    """
    feat_names = train_ds.feature_names
    method = 'SHAP LinearExplainer'

    try:
        import shap
        X_train = scaler.transform(train_ds.features)
        X_test  = scaler.transform(test_ds.features)
        explainer = shap.LinearExplainer(model, X_train,
                                         feature_perturbation='interventional')
        shap_vals = explainer.shap_values(X_test)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
    except Exception:
        # Ehtiyat: logistic regression koefficientleri
        mean_abs = np.abs(model.coef_[0])
        method   = 'LR Koefficientləri (XAI)'

    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    return {
        'features':   [_shorten(feat_names[i]) for i in top_idx],
        'importance': [round(float(mean_abs[i]), 4) for i in top_idx],
        'method':     method,
        'ok':         True
    }


def analyze_from_csvs(biased_df: pd.DataFrame, unbiased_df: pd.DataFrame):
    cfg = _detect_config(biased_df)

    b_ds = _csv_to_aif360(biased_df,   cfg)
    u_ds = _csv_to_aif360(unbiased_df, cfg)

    b_train, b_test = b_ds.split([0.7], shuffle=True, seed=42)
    u_train, u_test = u_ds.split([0.7], shuffle=True, seed=42)

    b_train, u_train = _align(b_train, u_train)
    b_test,  u_test  = _align(b_test,  u_test)

    b_acc, b_pred, b_model, b_scaler = _fit_predict(b_train, b_test)
    u_acc, u_pred, u_model, u_scaler = _fit_predict(u_train, u_test)

    biased   = _bias_metrics(b_test, b_pred, cfg)
    unbiased = _bias_metrics(u_test, u_pred, cfg)

    biased['accuracy']   = b_acc
    unbiased['accuracy'] = u_acc
    biased['config']     = cfg
    unbiased['config']   = cfg

    # SHAP
    biased['shap']   = _compute_shap(b_model, b_scaler, b_train, b_test)
    unbiased['shap'] = _compute_shap(u_model, u_scaler, u_train, u_test)

    del b_ds, u_ds, b_train, b_test, u_train, u_test, b_pred, u_pred
    gc.collect()

    return biased, unbiased
