"""
pipeline.py
-----------
Regression pipeline logic: column type detection, encoding, backward
elimination via statsmodels OLS, and the main prediction loop.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Column-type detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return a dict {col: 'numeric' | 'boolean' | 'categorical'} for every
    column that could be used as a feature (excludes object cols with too many
    unique values only when they're not boolean)."""
    result: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        unique_vals = s.dropna().unique()

        # Boolean-like: dtype bool OR exactly {0,1} / {True,False}
        if s.dtype == bool or set(unique_vals).issubset({0, 1, True, False, "0", "1"}):
            result[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            result[col] = "numeric"
        else:
            result[col] = "categorical"
    return result


def split_candidates(df: pd.DataFrame, targets: list[str], group_col: str) -> dict[str, list[str]]:
    """Return candidate feature columns split by type, excluding targets,
    constant columns (nunique == 1), and near-ID columns (nunique / nrows > 0.95).
    group_col is intentionally kept so the user can choose to include it as a feature."""
    exclude = set(targets)
    type_map = detect_column_types(df)
    n_rows = len(df)
    out: dict[str, list[str]] = {"numeric": [], "boolean": [], "categorical": []}
    for col, t in type_map.items():
        if col in exclude:
            continue
        n_unique = df[col].nunique()
        if n_unique <= 1:
            continue  # constant column
        if t == "categorical" and n_unique / n_rows > 0.95:
            continue  # near-ID column (e.g. sre code)
        out[t].append(col)
    for k in out:
        out[k] = sorted(out[k])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Boolean-merge helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_boolean_merge(
    df: pd.DataFrame,
    merges: dict[str, tuple[list, str]],  # {col: ([true_values], new_col_name)}
) -> pd.DataFrame:
    """
    Create new boolean columns from categorical columns.
    merges: {'situacao': (['DUP'], 'situacao_is_dup')}
    The original column is kept; split_candidates will list the new col as boolean.
    """
    df = df.copy()
    for col, (true_vals, new_name) in merges.items():
        if col in df.columns:
            df[new_name] = df[col].isin(true_vals).astype(bool)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering(
    df: pd.DataFrame,
    targets: list[str],
    group_col: str = "regional_macro",
) -> pd.DataFrame:
    """
    Append engineered features to *df* (original columns unchanged):

    1. **Log-transforms** of skewed numeric columns  → ``log_<col>``
    2. **Socioeconomic ratios** (per-capita, per-km)  → e.g. ``frotas_per_capita``
    3. **Log-transforms of ratio features**           → ``log_<ratio>``
    4. **Target-based location features** — mean and median of each target's
       observed values, grouped by ``go`` and ``group_col``, propagated to
       ALL rows (including unobserved ones).          → ``<target>_media_go``, etc.

    Groups for (4) are built from OBSERVED rows only, so unobserved SREs
    receive only the group-level reference value, not their own target.
    """
    df = df.copy()

    # ── 1. Log-transforms of base numeric features ────────────────────────────
    _log_base = [
        "distancia_cid_1", "distancia_cid_1_2_3",
        "media_pib", "media_populacao_residente",
        "media_frotas_ativas", "media_empresas_ativas",
        "extensao",
    ]
    for col in _log_base:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── 2. Socioeconomic ratio features ──────────────────────────────────────
    _pop = df["media_populacao_residente"] if "media_populacao_residente" in df.columns else None
    _ext = df["extensao"]                  if "extensao"                  in df.columns else None
    _pib = df["media_pib"]                 if "media_pib"                 in df.columns else None
    _fro = df["media_frotas_ativas"]       if "media_frotas_ativas"       in df.columns else None
    _emp = df["media_empresas_ativas"]     if "media_empresas_ativas"     in df.columns else None

    if _fro is not None and _pop is not None:
        df["frotas_per_capita"]   = _fro / _pop.replace(0, np.nan)
    if _pib is not None and _pop is not None:
        df["pib_per_capita"]      = _pib / _pop.replace(0, np.nan)
    if _emp is not None and _pop is not None:
        df["empresas_per_capita"] = _emp / _pop.replace(0, np.nan)
    if _fro is not None and _ext is not None:
        df["frotas_per_km"]       = _fro / _ext.replace(0, np.nan)
    if _emp is not None and _ext is not None:
        df["empresas_per_km"]     = _emp / _ext.replace(0, np.nan)
    if _pib is not None and _ext is not None:
        df["pib_per_km"]          = _pib / _ext.replace(0, np.nan)

    # ── 3. Log-transforms of ratio features ──────────────────────────────────
    _ratio_cols = [
        "frotas_per_capita", "pib_per_capita", "empresas_per_capita",
        "frotas_per_km", "empresas_per_km", "pib_per_km",
    ]
    for col in _ratio_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── 3b. Excess features (relative to regional average) ───────────────────
    # Captures whether an SRE has unusually high/low intensity for its region.
    if _fro is not None and _ext is not None and group_col in df.columns:
        reg_avg_frotas_km = df.groupby(group_col)["frotas_per_km"].transform("mean")
        df["excesso_frotas_km"] = df["frotas_per_km"] / reg_avg_frotas_km.clip(lower=1)
        df["log_excesso_frotas_km"] = np.log1p(df["excesso_frotas_km"].clip(lower=0))
    if _pib is not None and _ext is not None and group_col in df.columns:
        reg_avg_pib_km = df.groupby(group_col)["pib_per_km"].transform("mean")
        df["excesso_pib_km"] = df["pib_per_km"] / reg_avg_pib_km.clip(lower=1)
        df["log_excesso_pib_km"] = np.log1p(df["excesso_pib_km"].clip(lower=0))

    # ── 4. Target-based location features ────────────────────────────────────
    # Compute mean/median/CV of observed target values per group, then map to
    # ALL rows so unobserved SREs get a group-level reference.
    # Finally, combine them into a single composite feature per target via PCA.
    _loc_groups: dict[str, str] = {}
    if "go" in df.columns:
        _loc_groups["go"] = "go"
    if group_col in df.columns and group_col not in _loc_groups:
        suffix = "regional" if "regional" in group_col.lower() else group_col
        _loc_groups[group_col] = suffix

    for target in targets:
        if target not in df.columns:
            continue
        obs_mask = df[target].notna()
        if obs_mask.sum() == 0:
            continue

        # Build individual location features
        loc_cols = []
        for gcol, suffix in _loc_groups.items():
            stats = df.loc[obs_mask].groupby(gcol)[target].agg(["mean", "median", "std"])
            mean_col = f"{target}_media_{suffix}"
            med_col  = f"{target}_mediana_{suffix}"
            cv_col   = f"{target}_cv_{suffix}"

            df[mean_col] = df[gcol].map(stats["mean"])
            df[med_col]  = df[gcol].map(stats["median"])
            cv_series = stats["std"] / stats["mean"].clip(lower=1)
            df[cv_col] = df[gcol].map(cv_series)

            loc_cols.extend([mean_col, med_col, cv_col])

        # ── Combine into single composite via PCA ────────────────────────────
        if len(loc_cols) >= 2:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler as LocScaler

            # Standardize and run PCA on ALL rows (no leakage — individual
            # features are already computed from observed data only)
            X_loc = df[loc_cols].fillna(df[loc_cols].median()).values.astype(float)
            X_loc_scaled = LocScaler().fit_transform(X_loc)
            pca = PCA(n_components=1)
            df[f"localizacao_{target}"] = pca.fit_transform(X_loc_scaled)[:, 0]
            # Individual location columns are kept alongside the PCA composite
            # so the user can choose either approach in the Streamlit sidebar.

    return df


def normalize_group_label(v) -> str:
    """Normalize a group column value to a clean string label.
    Converts float whole numbers to int strings: 1.0 → '1', 2.0 → '2'.
    Non-whole floats and strings are returned as-is.
    """
    s = str(v)
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def apply_group_merge(
    df: pd.DataFrame,
    group_col: str,
    merges: list[tuple[list, str]],  # [([val1, val2], merged_label), ...]
) -> pd.DataFrame:
    """
    Replace values in group_col: each (values_list, merged_label) maps all
    listed values to merged_label.  Used to combine regions for regression.
    """
    if not merges:
        df = df.copy()
        df[group_col] = df[group_col].map(normalize_group_label)
        return df
    df = df.copy()
    # Normalise column first (1.0 → '1') so string comparisons are consistent
    df[group_col] = df[group_col].map(normalize_group_label)
    for values, label in merges:
        str_vals = [normalize_group_label(v) for v in values]
        mask = df[group_col].isin(str_vals)
        df.loc[mask, group_col] = label
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(
    df: pd.DataFrame,
    selected_features: list[str],
    encoding_choices: dict[str, str],  # col -> 'onehot' | 'label'
    fit_encoders: dict | None = None,   # pass fitted encoders to reuse on pred set
) -> tuple[pd.DataFrame, dict]:
    """
    Encode selected features and return:
        (encoded_df, encoders_dict)
    Numeric and boolean cols are returned as-is (cast to float).
    Categorical cols use label or one-hot encoding.
    If fit_encoders is provided, uses those instead of fitting new ones.
    """
    parts: list[pd.DataFrame] = []
    encoders: dict = fit_encoders.copy() if fit_encoders else {}

    type_map = detect_column_types(df)

    for col in selected_features:
        if col not in df.columns:
            continue
        col_type = type_map.get(col, "numeric")

        # If encoding_choices explicitly requests categorical treatment (onehot/label),
        # override the auto-detected type.  This is needed for group_col (regional_macro)
        # which may have a numeric dtype in the raw DataFrame but must be one-hot encoded.
        if encoding_choices.get(col) in ("onehot", "label") and col_type != "categorical":
            col_type = "categorical"

        if col_type in ("numeric", "boolean"):
            parts.append(df[[col]].astype(float).reset_index(drop=True))

        elif col_type == "categorical":
            choice = encoding_choices.get(col, "onehot")
            if choice == "label":
                if fit_encoders and col in fit_encoders:
                    le: LabelEncoder = fit_encoders[col]
                    known = set(le.classes_)
                    safe = df[col].apply(lambda v: v if v in known else le.classes_[0])
                    encoded = le.transform(safe.astype(str))
                else:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col].fillna("__missing__").astype(str))
                    encoders[col] = le
                parts.append(
                    pd.DataFrame({col: encoded.astype(float)}).reset_index(drop=True)
                )
            else:  # onehot
                dummies = pd.get_dummies(
                    df[col].fillna("__missing__").astype(str), prefix=col, drop_first=True
                ).astype(float)
                if fit_encoders and col in fit_encoders:
                    expected_cols = fit_encoders[col]
                    dummies = dummies.reindex(columns=expected_cols, fill_value=0.0)
                else:
                    encoders[col] = list(dummies.columns)
                dummies = dummies.reset_index(drop=True)
                parts.append(dummies)

    if not parts:
        return pd.DataFrame(index=range(len(df))), encoders

    return pd.concat(parts, axis=1), encoders


# ─────────────────────────────────────────────────────────────────────────────
# Backward elimination via statsmodels OLS
# ─────────────────────────────────────────────────────────────────────────────

def backward_elimination_ols(
    X_sc: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float,
    mandatory_features: list[str] | None = None,
) -> tuple[list[int], list[str], object | None]:
    """
    Iteratively remove the feature with the highest p-value above `threshold`.
    Features listed in `mandatory_features` (original or one-hot prefix names)
    are never removed, regardless of their p-value.
    Returns:
        kept_indices   - column indices in X_sc that survived
        kept_names     - corresponding feature names
        ols_result     - fitted statsmodels OLS result (or None if 0 features)
    """
    if not mandatory_features:
        mandatory_features = []

    # Resolve which encoded columns are mandatory.
    # Handles exact names (e.g. "km_inicial") and one-hot prefixes
    # (e.g. "classe" protects "classe_Radiais", "classe_Longitudinais", …).
    mandatory_set: set[str] = set()
    for col in feature_names:
        for mf in mandatory_features:
            if col == mf or col.startswith(mf + "_"):
                mandatory_set.add(col)

    kept = list(range(X_sc.shape[1]))

    while True:
        if not kept:
            return [], [], None

        X_iter = sm.add_constant(X_sc[:, kept], has_constant="add")
        try:
            res = sm.OLS(y, X_iter).fit()
        except Exception:
            return kept, [feature_names[i] for i in kept], None

        # p-values: index 0 is the constant, 1: are the features
        pvals = res.pvalues[1:]

        # Only non-mandatory features are candidates for removal
        removable = [
            i for i, idx in enumerate(kept)
            if feature_names[idx] not in mandatory_set
        ]

        if not removable:
            break  # all remaining features are mandatory

        pvals_removable = pvals[removable]
        max_pval_removable = float(pvals_removable.max())

        if max_pval_removable <= threshold:
            break  # all removable features are statistically significant

        # Remove the worst removable feature
        worst_in_removable = int(pvals_removable.argmax())
        worst_local = removable[worst_in_removable]
        kept.pop(worst_local)

    return kept, [feature_names[i] for i in kept], res


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_pipeline(
    df: pd.DataFrame,
    group_col: str,
    targets: list[str],
    features_per_target: dict[str, list[str]],
    encoding_choices: dict[str, str],
    pvalue_threshold: float,
    log_transform: bool = False,
    mandatory_per_target: dict[str, list[str]] | None = None,
    max_features: int | None = None,
    clip_predictions: bool = False,
    progress_callback=None,
) -> dict:
    """
    Full prediction pipeline.  One global OLS model is fitted per target
    using ALL SREs together; group_col (regional_macro) is included as a
    one-hot feature so regional effects are captured automatically.

    Per-region metrics are derived post-hoc from the global model's residuals
    on each region's training subset — used for visualisation only.

    Returns a dict with keys:
        df_result     - original df with predicted values filled in
        metrics_df    - one row per (target, region) with R², RMSE, RSE, …
        pvalues_df    - one row per (target, feature) with p-value  (region="global")
        coefs_df      - one row per (target, feature) with coefficient (region="global")
        clip_log      - list of dicts for clipped-negative predictions
        fonte_cols    - dict {target: Series}
    """
    df_result = df.copy()
    fonte_cols: dict[str, pd.Series] = {
        t: pd.Series(
            ["observado" if pd.notna(df[t].iloc[i]) else np.nan for i in range(len(df))],
            index=df.index,
        )
        for t in targets
    }

    model_log: list[dict] = []
    pvalues_rows: list[dict] = []
    coefs_rows: list[dict] = []
    equations_rows: list[dict] = []
    residuals_rows: list[dict] = []
    clip_log: list[dict] = []

    regions = sorted(df[group_col].dropna().unique().tolist(), key=str)

    total_steps = len(targets)
    step = 0

    for target in targets:
        step += 1
        if progress_callback:
            progress_callback(step / total_steps, f"Target: {target}")

        target_features  = [f for f in features_per_target.get(target, []) if f != target]
        mandatory_features = [f for f in (mandatory_per_target or {}).get(target, []) if f != target]

        # Encode ALL rows (all regions together)
        encoded_all, _encoders = encode_features(
            df_result.reset_index(drop=False),
            target_features,
            encoding_choices,
        )
        encoded_all.index = df_result.index

        feat_cols = list(encoded_all.columns)
        has_all_features = encoded_all.notna().all(axis=1)
        mask_train = df_result[target].notna() & has_all_features
        mask_pred  = df_result[target].isna()  & has_all_features
        # Rows with null target AND missing features — can never be OLS-predicted
        mask_missing_no_feat = df_result[target].isna() & ~has_all_features

        n_train_global = int(mask_train.sum())
        n_pred_global  = int(mask_pred.sum())
        n_any_missing  = int(df_result[target].isna().sum())

        # Helper: log one row per region with the same metodo/r2
        def _per_region_log(metodo, r2=None, extra=None):
            for _reg in regions:
                _mr = df_result[group_col] == _reg
                _nt = int((mask_train & _mr).sum())
                _np = int(((mask_pred | mask_missing_no_feat) & _mr).sum())
                row = _log_row(target, _reg, _nt, _np, metodo, r2, group_col)
                if extra:
                    row.update(extra)
                model_log.append(row)

        if n_any_missing == 0:
            _per_region_log("sem_vazios")
            continue

        if n_train_global < 2:
            global_mean = df_result.loc[df_result[target].notna(), target].mean()
            mask_all_missing = mask_pred | mask_missing_no_feat
            df_result.loc[mask_all_missing, target] = global_mean
            fonte_cols[target].loc[mask_all_missing] = "previsto_media_global"
            _per_region_log("media_global")
            continue

        # All missing rows have incomplete features → cannot use OLS, fall back to mean
        if n_pred_global == 0:
            global_mean = df_result.loc[df_result[target].notna(), target].mean()
            df_result.loc[mask_missing_no_feat, target] = global_mean
            fonte_cols[target].loc[mask_missing_no_feat] = "previsto_media_global"
            _per_region_log("media_global")
            continue

        X_train_raw = encoded_all.loc[mask_train].values.astype(float)
        y_train     = df_result.loc[mask_train, target].values.astype(float)
        X_pred_raw  = encoded_all.loc[mask_pred].values.astype(float)
        idx_pred    = df_result[mask_pred].index

        # Log-transform: regress on log(y), predict exp(ŷ) — always positive
        y_train_model = np.log(np.maximum(y_train, 1e-9)) if log_transform else y_train

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_pred_sc  = scaler.transform(X_pred_raw)

        # R² do modelo completo (antes do backward elimination) — informativo
        try:
            _res_full = sm.OLS(y_train_model, sm.add_constant(X_train_sc, has_constant="add")).fit()
            r2_full = round(float(_res_full.rsquared), 4)
        except Exception:
            r2_full = None

        kept_idx, kept_names, ols_res = backward_elimination_ols(
            X_train_sc, y_train_model, feat_cols, pvalue_threshold, mandatory_features
        )

        # ── Limit to max_features (keep highest |t-stat|, always preserve mandatory) ──
        if max_features and kept_idx and ols_res is not None and len(kept_idx) > max_features:
            mandatory_set_enc: set[str] = set()
            for mc in mandatory_features:
                for kn in kept_names:
                    if kn == mc or kn.startswith(mc + "_"):
                        mandatory_set_enc.add(kn)
            mandatory_kept = [i for i, n in zip(kept_idx, kept_names) if n in mandatory_set_enc]
            non_mandatory  = [(i, n, abs(float(ols_res.tvalues[j + 1])))
                               for j, (i, n) in enumerate(zip(kept_idx, kept_names))
                               if n not in mandatory_set_enc]
            non_mandatory.sort(key=lambda x: x[2], reverse=True)
            slots = max(0, max_features - len(mandatory_kept))
            kept_idx   = mandatory_kept + [x[0] for x in non_mandatory[:slots]]
            kept_names = [kept_names[kept_idx.index(i)] for i in kept_idx] if kept_idx else []
            # Re-fit OLS on the reduced feature set
            _X_red = sm.add_constant(X_train_sc[:, kept_idx], has_constant="add")
            try:
                ols_res = sm.OLS(y_train_model, _X_red).fit()
            except Exception:
                pass

        if not kept_idx or ols_res is None:
            global_mean = df_result.loc[df_result[target].notna(), target].mean()
            mask_all_missing = mask_pred | mask_missing_no_feat
            df_result.loc[mask_all_missing, target] = global_mean
            fonte_cols[target].loc[mask_all_missing] = "previsto_media_global"
            _per_region_log(
                "media_global_sem_features",
                extra={
                    "r2_modelo_completo": r2_full,
                    "aviso": f"Todas as features eliminadas (p>{pvalue_threshold}). R²_completo={r2_full}. Aumente o threshold.",
                },
            )
            continue

        X_train_final = X_train_sc[:, kept_idx]
        X_pred_final  = X_pred_sc[:, kept_idx]

        # Duan smearing factor — corrects retransformation bias when log_transform=True.
        # exp(ŷ_log) estimates the median of y, not the mean.
        # smearing = mean(exp(ε_i)) where ε_i = y_log_i - ŷ_log_i (OLS residuals in log space).
        if log_transform:
            _y_train_hat_log = ols_res.predict(sm.add_constant(X_train_final, has_constant="add"))
            _smearing = float(np.mean(np.exp(y_train_model - _y_train_hat_log)))
        else:
            _smearing = 1.0

        # Predict
        y_hat = ols_res.predict(sm.add_constant(X_pred_final, has_constant="add"))
        if log_transform:
            y_hat = np.exp(y_hat) * _smearing

        # Clip negatives (safety net; should not trigger with log_transform)
        n_neg = int((y_hat < 0).sum())
        if n_neg > 0:
            clip_log.append(
                {
                    "target": target,
                    group_col: "global",
                    "n_negativos_clampados": n_neg,
                    "min_previsto_bruto": round(float(y_hat.min()), 4),
                    "media_y_treino": round(float(y_train.mean()), 4),
                }
            )
        y_hat = np.maximum(y_hat, 0)

        df_result.loc[idx_pred, target] = np.round(y_hat, 4)
        fonte_cols[target].loc[idx_pred] = "previsto"

        # Fallback for rows with null target that had incomplete features
        # (could not be predicted by OLS) → fill with mean of observed training values
        if mask_missing_no_feat.sum() > 0:
            fallback_mean = round(float(y_train.mean()), 4)
            df_result.loc[mask_missing_no_feat, target] = fallback_mean
            fonte_cols[target].loc[mask_missing_no_feat] = "previsto_media_global"

        # Global model predictions on training set (for metrics)
        r2_global = float(ols_res.rsquared)
        X_train_ols        = sm.add_constant(X_train_final, has_constant="add")
        y_pred_train_model = ols_res.predict(X_train_ols)
        # Apply same smearing factor used in out-of-sample predictions
        y_pred_train = np.exp(y_pred_train_model) * _smearing if log_transform else y_pred_train_model
        p_global = X_train_final.shape[1]

        # Store fitted values so export_to_excel can build the ajustados_obs sheet
        df_result.loc[mask_train, f"{target}_ajustado"] = np.round(y_pred_train, 4)

        train_regions_arr = df_result.loc[mask_train, group_col].values
        pred_regions_arr  = df_result.loc[mask_pred,  group_col].values

        # ── Per-region metrics (from the global model's residuals) ────────────
        for regiao in regions:
            r_mask_tr = train_regions_arr == regiao
            r_mask_pr = pred_regions_arr  == regiao

            n_tr_reg = int(r_mask_tr.sum())
            n_pr_reg = int(r_mask_pr.sum())

            if n_tr_reg == 0:
                model_log.append(
                    _log_row(target, regiao, 0, n_pr_reg, "regressao_linear", None, group_col)
                )
                continue

            y_tr_reg  = y_train[r_mask_tr]
            y_pr_reg  = y_pred_train[r_mask_tr]
            resid_reg = y_tr_reg - y_pr_reg

            ss_res_reg = float(np.sum(resid_reg ** 2))
            ss_tot_reg = float(np.sum((y_tr_reg - y_tr_reg.mean()) ** 2))
            r2_reg = (1.0 - ss_res_reg / ss_tot_reg) if ss_tot_reg > 0 else float("nan")

            mse_reg  = float(np.mean(resid_reg ** 2))
            mae_reg  = float(np.mean(np.abs(resid_reg)))
            rmse_reg = math.sqrt(mse_reg)
            nonzero_reg = y_tr_reg != 0
            mape_reg = float(np.mean(np.abs(resid_reg[nonzero_reg] / y_tr_reg[nonzero_reg])) * 100) if nonzero_reg.any() else None
            denom_reg = n_tr_reg - p_global - 1
            rse_reg   = math.sqrt(ss_res_reg / denom_reg) if denom_reg > 0 else None

            model_log.append(
                {
                    **_log_row(target, regiao, n_tr_reg, n_pr_reg, "regressao_linear", round(r2_reg, 4), group_col),
                    "r2_global":          round(r2_global, 4),
                    "r2_modelo_completo": r2_full,
                    "mse":   round(mse_reg,  4),
                    "mae":   round(mae_reg,  4),
                    "rmse":  round(rmse_reg, 4),
                    "mape":  round(mape_reg, 4) if mape_reg is not None else None,
                    "rse":   round(rse_reg,  4) if rse_reg  is not None else None,
                    "features_usadas":   ", ".join(kept_names),
                    "n_features_usadas": len(kept_names),
                    "aviso": None,
                }
            )

            for _obs, _pred, _resid in zip(y_tr_reg, y_pr_reg, resid_reg):
                residuals_rows.append(
                    {
                        "target":  target,
                        group_col: regiao,
                        "y_obs":   round(float(_obs),   4),
                        "y_pred":  round(float(_pred),  4),
                        "residuo": round(float(_resid), 4),
                    }
                )

        # ── P-values and coefficients — one global model ──────────────────────
        for j, fname in enumerate(kept_names):
            pvalues_rows.append(
                {
                    "target":        target,
                    group_col:       "global",
                    "feature":       fname,
                    "coef":          round(float(ols_res.params[j + 1]), 6),
                    "pvalue":        round(float(ols_res.pvalues[j + 1]), 6),
                    "significativo": ols_res.pvalues[j + 1] <= pvalue_threshold,
                }
            )
            coefs_rows.append(
                {
                    "target":  target,
                    group_col: "global",
                    "feature": fname,
                    "coef":    round(float(ols_res.params[j + 1]), 6),
                }
            )

        # ── Build equation string ─────────────────────────────────────────────
        intercept = float(ols_res.params[0])
        padding   = " " * (len(target) + 5)
        lhs = f"log(ŷ({target}))" if log_transform else f"ŷ({target})"
        eq_parts = [f"{lhs} = {intercept:+.4f}"]
        for _j, _fname in enumerate(kept_names):
            _coef = float(ols_res.params[_j + 1])
            _sign = "+" if _coef >= 0 else "-"
            eq_parts.append(f"{padding}  {_sign} {abs(_coef):.4f} × {_fname}")
        if log_transform:
            eq_parts.append(
                f"\nŷ({target}) = exp(log(ŷ)) × {_smearing:.6f}"
                f"  [Duan smearing: mean(exp(ε_i)) calculado sobre {int(mask_train.sum())} obs.]"
            )
        equations_rows.append(
            {
                "target":       target,
                group_col:      "global",
                "intercept":    round(intercept, 4),
                "equation":     "\n".join(eq_parts),
                "n_features":   len(kept_names),
                "smearing":     round(_smearing, 6) if log_transform else None,
            }
        )

    # ── Clip predictions to [0.3×regional_median, 3.0×regional_median] ────────
    if clip_predictions:
        _regions_unique = sorted(df[group_col].dropna().unique().tolist(), key=str)
        for t in targets:
            obs_mask_orig = df[t].notna()  # original observed values (never clip)
            for reg in _regions_unique:
                reg_mask = df_result[group_col].astype(str) == str(reg)
                obs_reg = df.loc[obs_mask_orig & reg_mask, t]
                if len(obs_reg) == 0:
                    continue
                med = obs_reg.median()
                lo = 0.3 * med
                hi = 3.0 * med
                # Clip predicted rows
                pred_mask = reg_mask & ~obs_mask_orig
                df_result.loc[pred_mask, t] = df_result.loc[pred_mask, t].clip(lo, hi)
                # Clip fitted values for observed rows
                adj_col = f"{t}_ajustado"
                if adj_col in df_result.columns:
                    df_result.loc[obs_mask_orig & reg_mask, adj_col] = (
                        df_result.loc[obs_mask_orig & reg_mask, adj_col].clip(lo, hi)
                    )

    # ── Fonte columns ─────────────────────────────────────────────────────────
    for t in targets:
        df_result[f"fonte_{t}"] = fonte_cols[t]

    def _define_fonte(row):
        valores = [row.get(f"fonte_{t}") for t in targets]
        unicos = {v for v in valores if pd.notna(v)}
        if unicos == {"observado"}:
            return "observado"
        if unicos == {"previsto"}:
            return "previsto"
        if "previsto_media_global" in unicos and "observado" not in unicos:
            return "previsto_media_global"
        return "misto"

    df_result["fonte"] = df_result.apply(_define_fonte, axis=1)

    metrics_df   = pd.DataFrame(model_log)
    pvalues_df   = pd.DataFrame(pvalues_rows)   if pvalues_rows   else pd.DataFrame()
    coefs_df     = pd.DataFrame(coefs_rows)     if coefs_rows     else pd.DataFrame()
    equations_df = pd.DataFrame(equations_rows) if equations_rows else pd.DataFrame()
    residuals_df = pd.DataFrame(residuals_rows) if residuals_rows else pd.DataFrame()

    return {
        "df_result":     df_result,
        "metrics_df":    metrics_df,
        "pvalues_df":    pvalues_df,
        "coefs_df":      coefs_df,
        "equations_df":  equations_df,
        "residuals_df":  residuals_df,
        "clip_log":      clip_log,
        "fonte_cols":    fonte_cols,
        "group_col":     group_col,
        "log_transform": log_transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Imputation helpers  (média regional / KNN)
# ─────────────────────────────────────────────────────────────────────────────

def _impute_media_regional(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    log_transform: bool,
) -> tuple[pd.Series, list[dict], list[dict]]:
    """
    Fill missing target values with the (geometric) mean of observed values
    in the same regional_macro group.  LOO cross-validation is used to
    compute per-region metrics without data leakage.

    Returns (predicted_series, metrics_rows, residuals_rows).
    """
    obs_mask = df[target].notna()
    predicted = df[target].copy().astype(float)

    # Global fallback (used when a region has zero observations)
    y_global = df.loc[obs_mask, target].values.astype(float)
    if log_transform:
        valid_g = y_global > 0
        global_mean = float(np.exp(np.log(y_global[valid_g]).mean())) if valid_g.any() else float(y_global.mean() if len(y_global) else 0.0)
    else:
        global_mean = float(y_global.mean()) if len(y_global) else 0.0

    metrics_rows: list[dict] = []
    residuals_rows: list[dict] = []

    for region in sorted(df[group_col].dropna().unique(), key=str):
        reg_mask = df[group_col].astype(str) == str(region)
        obs_idx  = df.index[reg_mask & obs_mask].tolist()
        miss_idx = df.index[reg_mask & ~obs_mask].tolist()
        n_obs  = len(obs_idx)
        n_miss = len(miss_idx)

        if n_obs == 0:
            region_mean = global_mean
        else:
            y_r = df.loc[obs_idx, target].values.astype(float)
            if log_transform:
                valid = y_r > 0
                region_mean = float(np.exp(np.log(y_r[valid]).mean())) if valid.any() else global_mean
            else:
                region_mean = float(y_r.mean())

        if miss_idx:
            predicted.loc[miss_idx] = region_mean

        # LOO cross-validation for metrics (requires ≥ 2 observed in region)
        if n_obs >= 2:
            y_r = df.loc[obs_idx, target].values.astype(float)
            loo_preds = []
            for j in range(n_obs):
                others = np.array([y_r[k] for k in range(n_obs) if k != j])
                if log_transform:
                    valid = others > 0
                    loo_preds.append(float(np.exp(np.log(others[valid]).mean())) if valid.any() else float(others.mean()))
                else:
                    loo_preds.append(float(others.mean()))

            y_pred_loo = np.array(loo_preds)
            resid = y_r - y_pred_loo
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y_r - y_r.mean()) ** 2))
            r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse  = float(np.sqrt(np.mean(resid ** 2)))
            mae   = float(np.mean(np.abs(resid)))
            mse   = float(np.mean(resid ** 2))
            nz    = y_r != 0
            mape  = float(np.mean(np.abs(resid[nz] / y_r[nz])) * 100) if nz.any() else None

            metrics_rows.append({
                "target": target, group_col: str(region),
                "metodo": "media_regional",
                "n_train": n_obs, "n_pred": n_miss,
                "r2": round(r2, 4), "r2_global": None,  # filled after loop
                "rmse": round(rmse, 4), "mae": round(mae, 4),
                "mse": round(mse, 4),
                "mape": round(mape, 4) if mape is not None else None,
            })
            for y_o, y_p, r in zip(y_r, y_pred_loo, resid):
                residuals_rows.append({
                    "target": target, group_col: str(region),
                    "y_obs":   round(float(y_o), 4),
                    "y_pred":  round(float(y_p), 4),
                    "residuo": round(float(r),   4),
                })
        else:
            metrics_rows.append({
                "target": target, group_col: str(region),
                "metodo": "media_regional",
                "n_train": n_obs, "n_pred": n_miss,
                "r2": None, "r2_global": None,
                "rmse": None, "mae": None, "mse": None, "mape": None,
            })

    # Compute global LOO R² across all residuals for this target
    all_r = [row for row in residuals_rows if row["target"] == target]
    if all_r:
        y_obs_all   = np.array([row["y_obs"]   for row in all_r])
        resid_all   = np.array([row["residuo"] for row in all_r])
        ss_res_all  = float(np.sum(resid_all ** 2))
        ss_tot_all  = float(np.sum((y_obs_all - y_obs_all.mean()) ** 2))
        r2_global   = float(1 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0.0
    else:
        r2_global = None

    for row in metrics_rows:
        if row["target"] == target:
            row["r2_global"] = round(r2_global, 4) if r2_global is not None else None

    return predicted, metrics_rows, residuals_rows


def _impute_knn(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    target_features: list[str],
    encoding_choices: dict[str, str],
    knn_k: int,
    log_transform: bool,
) -> tuple[pd.Series, list[dict], list[dict]]:
    """
    Fill missing target values using a KNeighborsRegressor trained on
    the encoded features of observed rows.  KFold cross-validation on
    observed rows provides per-region metrics.

    Falls back to _impute_media_regional when no features are provided
    or when there are fewer than 2 observed rows.

    Returns (predicted_series, metrics_rows, residuals_rows).
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold

    obs_mask  = df[target].notna()
    miss_mask = ~obs_mask
    predicted = df[target].copy().astype(float)

    if not target_features:
        return _impute_media_regional(df, target, group_col, log_transform)

    # Encode ALL rows (observed + missing) so we can predict the missing ones
    encoded_all, _enc = encode_features(
        df.reset_index(drop=False), target_features, encoding_choices
    )
    encoded_all.index = df.index

    has_features = encoded_all.notna().all(axis=1)
    mask_train   = obs_mask  & has_features
    mask_pred    = miss_mask & has_features

    X_obs      = encoded_all.loc[mask_train].values.astype(float)
    y_obs_raw  = df.loc[mask_train, target].values.astype(float)
    X_miss     = encoded_all.loc[mask_pred].values.astype(float)
    obs_idx    = df.index[mask_train].tolist()

    if len(X_obs) < 2:
        return _impute_media_regional(df, target, group_col, log_transform)

    # Optionally log-transform y before fitting KNN
    if log_transform:
        valid = y_obs_raw > 0
        y_train_t = np.where(valid, np.log(y_obs_raw), np.nan)
        keep = ~np.isnan(y_train_t)
        X_train   = X_obs[keep]
        y_train   = y_train_t[keep]
        y_obs_met = y_obs_raw[keep]   # original scale for metrics
        obs_idx   = [obs_idx[i] for i, k in enumerate(keep) if k]
    else:
        X_train   = X_obs
        y_train   = y_obs_raw
        y_obs_met = y_obs_raw

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    k           = max(1, min(knn_k, len(X_train) - 1))

    # KFold CV predictions on observed rows
    n_splits   = min(5, len(X_train))
    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_preds_t = np.zeros(len(y_train))
    for tr_cv, va_cv in kf.split(X_train_sc):
        k_cv = max(1, min(k, len(tr_cv)))
        knn_cv = KNeighborsRegressor(n_neighbors=k_cv, weights="distance")
        knn_cv.fit(X_train_sc[tr_cv], y_train[tr_cv])
        cv_preds_t[va_cv] = knn_cv.predict(X_train_sc[va_cv])

    cv_preds = np.exp(cv_preds_t) if log_transform else cv_preds_t
    resid_all = y_obs_met - cv_preds

    # Global R²
    ss_res_g  = float(np.sum(resid_all ** 2))
    ss_tot_g  = float(np.sum((y_obs_met - y_obs_met.mean()) ** 2))
    r2_global = float(1 - ss_res_g / ss_tot_g) if ss_tot_g > 0 else 0.0

    # Per-region metrics from CV predictions
    groups_arr = df.loc[obs_idx, group_col].astype(str).values
    metrics_rows:   list[dict] = []
    residuals_rows: list[dict] = []

    for region in sorted(np.unique(groups_arr), key=str):
        rm = groups_arr == str(region)
        y_r = y_obs_met[rm]
        p_r = cv_preds[rm]
        r_r = y_r - p_r
        n_r = int(rm.sum())
        n_miss_r = int(((df[group_col].astype(str) == str(region)) & miss_mask).sum())

        if n_r < 2:
            metrics_rows.append({
                "target": target, group_col: str(region),
                "metodo": "knn", "n_train": n_r, "n_pred": n_miss_r,
                "r2": None, "r2_global": round(r2_global, 4),
                "rmse": None, "mae": None, "mse": None, "mape": None,
            })
            continue

        ss_res_r = float(np.sum(r_r ** 2))
        ss_tot_r = float(np.sum((y_r - y_r.mean()) ** 2))
        r2_r  = float(1 - ss_res_r / ss_tot_r) if ss_tot_r > 0 else 0.0
        rmse_r = float(np.sqrt(np.mean(r_r ** 2)))
        mae_r  = float(np.mean(np.abs(r_r)))
        mse_r  = float(np.mean(r_r ** 2))
        nz_r   = y_r != 0
        mape_r = float(np.mean(np.abs(r_r[nz_r] / y_r[nz_r])) * 100) if nz_r.any() else None

        metrics_rows.append({
            "target": target, group_col: str(region),
            "metodo": "knn", "n_train": n_r, "n_pred": n_miss_r,
            "r2": round(r2_r, 4), "r2_global": round(r2_global, 4),
            "rmse": round(rmse_r, 4), "mae": round(mae_r, 4),
            "mse": round(mse_r, 4),
            "mape": round(mape_r, 4) if mape_r is not None else None,
        })
        for y_o, y_p, r in zip(y_r, p_r, r_r):
            residuals_rows.append({
                "target": target, group_col: str(region),
                "y_obs":   round(float(y_o), 4),
                "y_pred":  round(float(y_p), 4),
                "residuo": round(float(r),   4),
            })

    # Fit full model on ALL observed rows and predict missing values
    knn_full = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn_full.fit(X_train_sc, y_train)
    if len(X_miss) > 0:
        X_miss_sc  = scaler.transform(X_miss)
        y_miss_t   = knn_full.predict(X_miss_sc)
        y_miss     = np.exp(y_miss_t) if log_transform else y_miss_t
        predicted.loc[mask_pred] = np.round(np.maximum(y_miss, 0), 4)

    return predicted, metrics_rows, residuals_rows


# ─────────────────────────────────────────────────────────────────────────────
# Imputation pipeline  (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_imputation_pipeline(
    df: pd.DataFrame,
    group_col: str,
    targets: list[str],
    method: str,                              # "media_regional" | "knn"
    features_per_target: dict[str, list[str]] | None = None,
    encoding_choices: dict[str, str] | None = None,
    knn_k: int = 5,
    log_transform: bool = False,
    progress_callback=None,
) -> dict:
    """
    Fill missing target values using regional mean or KNN from observed rows.
    Avoids overfitting by never fitting a parametric model — predictions are
    derived entirely from the observed target values in the same region or
    from the K nearest observed neighbours in feature space.

    Returns the same dict structure as run_regression_pipeline so all
    existing tabs and charts continue to work.
    pvalues_df / coefs_df / equations_df are returned as empty DataFrames.
    """
    df_result    = df.copy()
    all_metrics:   list[dict] = []
    all_residuals: list[dict] = []

    total = len(targets)
    for i, target in enumerate(targets):
        if progress_callback:
            progress_callback(i / total, f"Imputando {target} ({method})…")

        if method == "media_regional":
            predicted, mrows, rrows = _impute_media_regional(
                df, target, group_col, log_transform
            )
        elif method == "knn":
            tfeats = (features_per_target or {}).get(target, [])
            enc    = encoding_choices or {}
            predicted, mrows, rrows = _impute_knn(
                df, target, group_col, tfeats, enc, knn_k, log_transform
            )
        else:
            raise ValueError(f"Método desconhecido: {method!r}")

        df_result[target] = predicted
        all_metrics.extend(mrows)
        all_residuals.extend(rrows)

    if progress_callback:
        progress_callback(1.0, "Concluído! ✅")

    # Build fonte columns
    fonte_cols: dict[str, pd.Series] = {}
    for t in targets:
        s = pd.Series(index=df.index, dtype=object)
        s[df[t].notna()] = "observado"
        s[df[t].isna()]  = "previsto"
        s[df_result[t].isna()] = np.nan   # rows where even imputation failed
        df_result[f"fonte_{t}"] = s
        fonte_cols[t] = s

    def _define_fonte(row):
        vals   = [row.get(f"fonte_{t}") for t in targets]
        unique = {v for v in vals if pd.notna(v)}
        if unique == {"observado"}:     return "observado"
        if unique == {"previsto"}:      return "previsto"
        return "misto"

    df_result["fonte"] = df_result.apply(_define_fonte, axis=1)

    return {
        "df_result":     df_result,
        "metrics_df":    pd.DataFrame(all_metrics)    if all_metrics    else pd.DataFrame(),
        "pvalues_df":    pd.DataFrame(),
        "coefs_df":      pd.DataFrame(),
        "equations_df":  pd.DataFrame(),
        "residuals_df":  pd.DataFrame(all_residuals) if all_residuals else pd.DataFrame(),
        "clip_log":      [],
        "fonte_cols":    fonte_cols,
        "group_col":     group_col,
        "log_transform": log_transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stratified Regression Pipeline  (one OLS per region)
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_pipeline_stratified(
    df: pd.DataFrame,
    group_col: str,
    targets: list[str],
    features_per_target: dict[str, list[str]],
    encoding_choices: dict[str, str],
    pvalue_threshold: float,
    log_transform: bool = False,
    mandatory_per_target: dict[str, list[str]] | None = None,
    max_features: int | None = None,
    min_train_region: int = 10,
    clip_predictions: bool = False,
    progress_callback=None,
) -> dict:
    """
    Stratified pipeline: one OLS model is fitted **per region** for each target.
    Regions with fewer than *min_train_region* observed rows fall back to the
    regional mean (geometric mean if log_transform=True).

    This allows feature coefficients to vary across regions, capturing regional
    heterogeneity that a single global model cannot.

    Returns the same dict structure as run_regression_pipeline so all tabs
    and charts continue to work.
    """
    df_result = df.copy()
    fonte_cols: dict[str, pd.Series] = {
        t: pd.Series(
            ["observado" if pd.notna(df[t].iloc[i]) else np.nan for i in range(len(df))],
            index=df.index,
        )
        for t in targets
    }

    model_log: list[dict] = []
    pvalues_rows: list[dict] = []
    coefs_rows: list[dict] = []
    equations_rows: list[dict] = []
    residuals_rows: list[dict] = []
    clip_log: list[dict] = []

    regions = sorted(df[group_col].dropna().unique().tolist(), key=str)
    total_steps = len(targets) * len(regions)
    step = 0

    for target in targets:
        target_features = [f for f in features_per_target.get(target, []) if f != target]
        mandatory_features = [f for f in (mandatory_per_target or {}).get(target, []) if f != target]

        # Encode ALL rows once
        encoded_all, _encoders = encode_features(
            df_result.reset_index(drop=False),
            target_features,
            encoding_choices,
        )
        encoded_all.index = df_result.index
        feat_cols_all = list(encoded_all.columns)

        # Fallback: regional mean for regions with too few observations
        obs_mask_all = df_result[target].notna()
        y_global = df_result.loc[obs_mask_all, target].values.astype(float)
        if log_transform:
            valid_g = y_global > 0
            global_mean = float(np.exp(np.log(y_global[valid_g]).mean())) if valid_g.any() else float(y_global.mean())
        else:
            global_mean = float(y_global.mean()) if len(y_global) else 0.0

        # Per-region processing
        for regiao in regions:
            step += 1
            if progress_callback:
                progress_callback(step / total_steps, f"Target: {target} | Região: {regiao}")

            reg_mask = df_result[group_col].astype(str) == str(regiao)
            obs_mask = df_result[target].notna() & reg_mask
            miss_mask = df_result[target].isna() & reg_mask
            has_features = encoded_all.notna().all(axis=1)
            obs_train = obs_mask & has_features
            miss_pred = miss_mask & has_features
            miss_nofeat = miss_mask & ~has_features

            n_train = int(obs_train.sum())
            n_pred = int(miss_pred.sum())
            n_nofeat = int(miss_nofeat.sum())

            if n_train == 0 and n_pred == 0 and n_nofeat == 0:
                model_log.append(_log_row(target, regiao, 0, 0, "sem_dados", None, group_col))
                continue

            # ── Fallback: too few training rows ──────────────────────────
            if n_train < min_train_region:
                y_reg = df_result.loc[obs_mask, target].values.astype(float)
                if log_transform and len(y_reg) > 0:
                    valid = y_reg > 0
                    reg_mean = float(np.exp(np.log(y_reg[valid]).mean())) if valid.any() else global_mean
                elif len(y_reg) > 0:
                    reg_mean = float(y_reg.mean())
                else:
                    reg_mean = global_mean

                all_miss = miss_pred | miss_nofeat
                df_result.loc[all_miss, target] = round(reg_mean, 4)
                fonte_cols[target].loc[all_miss] = "previsto_media_regional"

                # LOO metrics for the fallback (if possible)
                if n_train >= 2:
                    y_r = df_result.loc[obs_train, target].values.astype(float)
                    loo_preds = []
                    for j in range(n_train):
                        others = np.array([y_r[k] for k in range(n_train) if k != j])
                        if log_transform:
                            v = others[others > 0]
                            loo_preds.append(float(np.exp(np.log(v).mean())) if len(v) > 0 else float(others.mean()))
                        else:
                            loo_preds.append(float(others.mean()))
                    y_pred_loo = np.array(loo_preds)
                    resid = y_r - y_pred_loo
                    ss_res = float(np.sum(resid ** 2))
                    ss_tot = float(np.sum((y_r - y_r.mean()) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
                    rmse = float(np.sqrt(np.mean(resid ** 2)))
                    mae = float(np.mean(np.abs(resid)))
                    mse = float(np.mean(resid ** 2))
                    nz = y_r != 0
                    mape = float(np.mean(np.abs(resid[nz] / y_r[nz])) * 100) if nz.any() else None
                    for obs, pred, r in zip(y_r, y_pred_loo, resid):
                        residuals_rows.append({
                            "target": target, group_col: str(regiao),
                            "y_obs": round(float(obs), 4),
                            "y_pred": round(float(pred), 4),
                            "residuo": round(float(r), 4),
                        })
                else:
                    r2 = rmse = mae = mse = mape = None

                model_log.append({
                    **_log_row(target, regiao, n_train, n_pred + n_nofeat,
                              "media_regional_fallback", round(r2, 4) if r2 is not None else None, group_col),
                    "r2_global": round(r2, 4) if r2 is not None else None,
                    "r2_modelo_completo": None,
                    "mse": round(mse, 4) if mse is not None else None,
                    "mae": round(mae, 4) if mae is not None else None,
                    "rmse": round(rmse, 4) if rmse is not None else None,
                    "mape": round(mape, 4) if mape is not None else None,
                    "rse": None,
                    "features_usadas": "",
                    "n_features_usadas": 0,
                    "aviso": f"Região com apenas {n_train} obs — fallback para média regional.",
                })
                continue

            # ── Full OLS for this region ─────────────────────────────────
            X_train_raw = encoded_all.loc[obs_train].values.astype(float)
            y_train = df_result.loc[obs_train, target].values.astype(float)
            X_pred_raw = encoded_all.loc[miss_pred].values.astype(float) if n_pred > 0 else np.empty((0, len(feat_cols_all)))
            idx_pred = df_result[miss_pred].index

            y_train_model = np.log(np.maximum(y_train, 1e-9)) if log_transform else y_train

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)

            # R² modelo completo
            try:
                _res_full = sm.OLS(y_train_model, sm.add_constant(X_train_sc, has_constant="add")).fit()
                r2_full = round(float(_res_full.rsquared), 4)
            except Exception:
                r2_full = None

            kept_idx, kept_names, ols_res = backward_elimination_ols(
                X_train_sc, y_train_model, feat_cols_all, pvalue_threshold, mandatory_features
            )

            if max_features and kept_idx and ols_res is not None and len(kept_idx) > max_features:
                mandatory_set_enc: set[str] = set()
                for mc in mandatory_features:
                    for kn in kept_names:
                        if kn == mc or kn.startswith(mc + "_"):
                            mandatory_set_enc.add(kn)
                mandatory_kept = [i for i, n in zip(kept_idx, kept_names) if n in mandatory_set_enc]
                non_mandatory = [(i, n, abs(float(ols_res.tvalues[j + 1])))
                                 for j, (i, n) in enumerate(zip(kept_idx, kept_names))
                                 if n not in mandatory_set_enc]
                non_mandatory.sort(key=lambda x: x[2], reverse=True)
                slots = max(0, max_features - len(mandatory_kept))
                kept_idx = mandatory_kept + [x[0] for x in non_mandatory[:slots]]
                kept_names = [kept_names[kept_idx.index(i)] for i in kept_idx] if kept_idx else []
                _X_red = sm.add_constant(X_train_sc[:, kept_idx], has_constant="add")
                try:
                    ols_res = sm.OLS(y_train_model, _X_red).fit()
                except Exception:
                    pass

            if not kept_idx or ols_res is None:
                y_reg = df_result.loc[obs_mask, target].values.astype(float)
                reg_mean = float(y_reg.mean()) if len(y_reg) else global_mean
                all_miss = miss_pred | miss_nofeat
                df_result.loc[all_miss, target] = round(reg_mean, 4)
                fonte_cols[target].loc[all_miss] = "previsto_media_regional"
                model_log.append({
                    **_log_row(target, regiao, n_train, n_pred + n_nofeat,
                              "media_regional_sem_features", None, group_col),
                    "r2_global": None, "r2_modelo_completo": r2_full,
                    "mse": None, "mae": None, "rmse": None, "mape": None, "rse": None,
                    "features_usadas": "", "n_features_usadas": 0,
                    "aviso": f"Backward eliminou todas as features (p>{pvalue_threshold}).",
                })
                continue

            X_train_final = X_train_sc[:, kept_idx]
            r2_reg_model = float(ols_res.rsquared)
            p_reg = X_train_final.shape[1]

            # Duan smearing
            if log_transform:
                _y_train_hat_log = ols_res.predict(sm.add_constant(X_train_final, has_constant="add"))
                _smearing = float(np.mean(np.exp(y_train_model - _y_train_hat_log)))
            else:
                _smearing = 1.0

            # Predict missing rows for this region
            if n_pred > 0:
                X_pred_sc = scaler.transform(X_pred_raw)
                X_pred_final = X_pred_sc[:, kept_idx]
                y_hat = ols_res.predict(sm.add_constant(X_pred_final, has_constant="add"))
                if log_transform:
                    y_hat = np.exp(y_hat) * _smearing
                y_hat = np.maximum(y_hat, 0)
                df_result.loc[idx_pred, target] = np.round(y_hat, 4)
                fonte_cols[target].loc[idx_pred] = "previsto"

            # Fallback for rows without features
            if n_nofeat > 0:
                y_reg = df_result.loc[obs_mask, target].values.astype(float)
                reg_mean = float(y_reg.mean()) if len(y_reg) else global_mean
                df_result.loc[miss_nofeat, target] = round(reg_mean, 4)
                fonte_cols[target].loc[miss_nofeat] = "previsto_media_regional"

            # Fitted values on training set (for metrics and export)
            y_pred_train_model = ols_res.predict(sm.add_constant(X_train_final, has_constant="add"))
            y_pred_train = np.exp(y_pred_train_model) * _smearing if log_transform else y_pred_train_model
            df_result.loc[obs_train, f"{target}_ajustado"] = np.round(y_pred_train, 4)

            # ── Metrics for this region ──────────────────────────────────
            resid_reg = y_train - y_pred_train
            ss_res_reg = float(np.sum(resid_reg ** 2))
            ss_tot_reg = float(np.sum((y_train - y_train.mean()) ** 2))
            r2_reg = (1.0 - ss_res_reg / ss_tot_reg) if ss_tot_reg > 0 else float("nan")
            mse_reg = float(np.mean(resid_reg ** 2))
            mae_reg = float(np.mean(np.abs(resid_reg)))
            rmse_reg = math.sqrt(mse_reg)
            nz = y_train != 0
            mape_reg = float(np.mean(np.abs(resid_reg[nz] / y_train[nz])) * 100) if nz.any() else None
            denom_reg = n_train - p_reg - 1
            rse_reg = math.sqrt(ss_res_reg / denom_reg) if denom_reg > 0 else None

            model_log.append({
                **_log_row(target, regiao, n_train, n_pred + n_nofeat,
                          "regressao_linear_strat", round(r2_reg, 4), group_col),
                "r2_global": round(r2_reg_model, 4),
                "r2_modelo_completo": r2_full,
                "mse": round(mse_reg, 4),
                "mae": round(mae_reg, 4),
                "rmse": round(rmse_reg, 4),
                "mape": round(mape_reg, 4) if mape_reg is not None else None,
                "rse": round(rse_reg, 4) if rse_reg is not None else None,
                "features_usadas": ", ".join(kept_names),
                "n_features_usadas": len(kept_names),
                "aviso": None,
            })

            for obs, pred, r in zip(y_train, y_pred_train, resid_reg):
                residuals_rows.append({
                    "target": target, group_col: str(regiao),
                    "y_obs": round(float(obs), 4),
                    "y_pred": round(float(pred), 4),
                    "residuo": round(float(r), 4),
                })

            # ── P-values and coefficients (region-specific) ──────────────
            for j, fname in enumerate(kept_names):
                pvalues_rows.append({
                    "target": target, group_col: str(regiao),
                    "feature": fname,
                    "coef": round(float(ols_res.params[j + 1]), 6),
                    "pvalue": round(float(ols_res.pvalues[j + 1]), 6),
                    "significativo": ols_res.pvalues[j + 1] <= pvalue_threshold,
                })
                coefs_rows.append({
                    "target": target, group_col: str(regiao),
                    "feature": fname,
                    "coef": round(float(ols_res.params[j + 1]), 6),
                })

            # ── Equation ─────────────────────────────────────────────────
            intercept = float(ols_res.params[0])
            padding = " " * (len(target) + len(str(regiao)) + 7)
            lhs = f"log(ŷ({target}))" if log_transform else f"ŷ({target})"
            eq_parts = [f"[Região {regiao}] {lhs} = {intercept:+.4f}"]
            for _j, _fname in enumerate(kept_names):
                _coef = float(ols_res.params[_j + 1])
                _sign = "+" if _coef >= 0 else "-"
                eq_parts.append(f"{padding}  {_sign} {abs(_coef):.4f} × {_fname}")
            if log_transform:
                eq_parts.append(
                    f"\nŷ({target}) = exp(log(ŷ)) × {_smearing:.6f}"
                    f"  [Duan smearing: mean(exp(ε_i)) sobre {n_train} obs.]"
                )
            equations_rows.append({
                "target": target, group_col: str(regiao),
                "intercept": round(intercept, 4),
                "equation": "\n".join(eq_parts),
                "n_features": len(kept_names),
                "smearing": round(_smearing, 6) if log_transform else None,
            })

    # ── Clip predictions to [0.3×regional_median, 3.0×regional_median] ────────
    if clip_predictions:
        _regions_unique = sorted(df[group_col].dropna().unique().tolist(), key=str)
        for t in targets:
            obs_mask_orig = df[t].notna()  # original observed values (never clip)
            for reg in _regions_unique:
                reg_mask = df_result[group_col].astype(str) == str(reg)
                obs_reg = df.loc[obs_mask_orig & reg_mask, t]
                if len(obs_reg) == 0:
                    continue
                med = obs_reg.median()
                lo = 0.3 * med
                hi = 3.0 * med
                # Clip predicted rows
                pred_mask = reg_mask & ~obs_mask_orig
                df_result.loc[pred_mask, t] = df_result.loc[pred_mask, t].clip(lo, hi)
                # Clip fitted values for observed rows
                adj_col = f"{t}_ajustado"
                if adj_col in df_result.columns:
                    df_result.loc[obs_mask_orig & reg_mask, adj_col] = (
                        df_result.loc[obs_mask_orig & reg_mask, adj_col].clip(lo, hi)
                    )

    # ── Fonte columns ─────────────────────────────────────────────────────
    for t in targets:
        df_result[f"fonte_{t}"] = fonte_cols[t]

    def _define_fonte(row):
        valores = [row.get(f"fonte_{t}") for t in targets]
        unicos = {v for v in valores if pd.notna(v)}
        if unicos == {"observado"}:
            return "observado"
        if unicos == {"previsto"}:
            return "previsto"
        if "previsto_media_regional" in unicos and "observado" not in unicos:
            return "previsto_media_regional"
        if "previsto_media_global" in unicos and "observado" not in unicos:
            return "previsto_media_global"
        return "misto"

    df_result["fonte"] = df_result.apply(_define_fonte, axis=1)

    metrics_df = pd.DataFrame(model_log)
    pvalues_df = pd.DataFrame(pvalues_rows) if pvalues_rows else pd.DataFrame()
    coefs_df = pd.DataFrame(coefs_rows) if coefs_rows else pd.DataFrame()
    equations_df = pd.DataFrame(equations_rows) if equations_rows else pd.DataFrame()
    residuals_df = pd.DataFrame(residuals_rows) if residuals_rows else pd.DataFrame()

    return {
        "df_result": df_result,
        "metrics_df": metrics_df,
        "pvalues_df": pvalues_df,
        "coefs_df": coefs_df,
        "equations_df": equations_df,
        "residuals_df": residuals_df,
        "clip_log": clip_log,
        "fonte_cols": fonte_cols,
        "group_col": group_col,
        "log_transform": log_transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(result: dict) -> bytes:
    """Serialize the pipeline result to an in-memory Excel workbook and return
    the raw bytes (ready to pass to st.download_button)."""
    import io, re

    df_full    = result["df_result"].copy()
    metrics_df = result["metrics_df"]
    clip_log   = result["clip_log"]
    targets    = list(result.get("fonte_cols", {}).keys())

    # Identify ajustado helper columns (written by run_regression_pipeline)
    ajust_cols = [f"{t}_ajustado" for t in targets if f"{t}_ajustado" in df_full.columns]

    # Drop per-target fonte_* helper columns — keep the unified 'fonte' column
    fonte_to_drop = [c for c in df_full.columns if c.startswith("fonte_")]
    df_full.drop(columns=fonte_to_drop, inplace=True)

    # Simplify 'fonte': anything not 'observado' becomes 'previsto'
    if "fonte" in df_full.columns:
        df_full["fonte"] = df_full["fonte"].apply(
            lambda v: "observado" if v == "observado" else "previsto"
        )

    # Drop booleanised helper columns  (<original>_is_<values>)
    bool_pattern = re.compile(r".+_is_.+")
    bool_cols = [c for c in df_full.columns
                 if bool_pattern.fullmatch(c) and c not in ajust_cols]
    df_full.drop(columns=bool_cols, inplace=True)

    # ─ Sheet 1: "dados" — full dataset without ajustado helpers ──────────────
    df_dados = df_full.drop(columns=[c for c in ajust_cols if c in df_full.columns])

    # ─ Sheet 2: "observados" — rows where all targets were originally observed ─
    df_obs = df_dados[df_dados["fonte"] == "observado"].copy()

    # ─ Sheet 3: "ajustados_obs" — rows with at least one OLS fitted value ─────
    # Uses rows where ANY {target}_ajustado is not null, so "misto" rows
    # (some targets observed, others not) are also included correctly.
    df_ajust = pd.DataFrame()
    if ajust_cols:  # only when regression ran (not imputation-only pipeline)
        has_fitted = df_full[ajust_cols].notna().any(axis=1)
        df_base = df_full[has_fitted].copy()

        # Non-target, non-ajustado columns as the base
        base_cols = [c for c in df_base.columns if c not in targets and c not in ajust_cols]
        df_ajust = df_base[base_cols].copy()

        # Add paired _observado / _ajustado / _residuo / _erro_pct for each target
        for t in targets:
            adj_col = f"{t}_ajustado"
            obs_col = f"{t}_observado"
            if t in df_base.columns:
                df_ajust[obs_col] = df_base[t].values
            if adj_col in df_base.columns:
                df_ajust[adj_col] = df_base[adj_col].values
                # Residual and percentage error (only where both values exist)
                if obs_col in df_ajust.columns:
                    obs_s = df_ajust[obs_col]
                    adj_s = df_ajust[adj_col]
                    df_ajust[f"{t}_residuo"]   = (obs_s - adj_s).round(4)
                    df_ajust[f"{t}_erro_pct"]  = (
                        ((obs_s - adj_s) / obs_s.replace(0, np.nan) * 100)
                        .round(2)
                    )

        # ── Confidence flag per target (based on regional P10/P90) ─────────
        group_col = result.get("group_col", "regional_macro")
        if group_col in df_ajust.columns:
            for t in targets:
                obs_col = f"{t}_observado"
                adj_col = f"{t}_ajustado"
                if obs_col not in df_ajust.columns or adj_col not in df_ajust.columns:
                    continue
                conf_col = f"{t}_confianca"
                df_ajust[conf_col] = "alta"
                for reg in df_ajust[group_col].dropna().unique():
                    reg_mask = df_ajust[group_col].astype(str) == str(reg)
                    obs_reg = df_ajust.loc[reg_mask, obs_col].dropna()
                    if len(obs_reg) < 5:
                        continue
                    p10 = np.percentile(obs_reg, 10)
                    p90 = np.percentile(obs_reg, 90)
                    adj_reg = df_ajust.loc[reg_mask, adj_col]
                    outside = (adj_reg < p10) | (adj_reg > p90)
                    df_ajust.loc[reg_mask & outside, conf_col] = "baixa"

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_dados.to_excel(writer, sheet_name="dados", index=False)
        df_obs.to_excel(writer, sheet_name="observados", index=False)
        if not df_ajust.empty:
            df_ajust.to_excel(writer, sheet_name="ajustados_obs", index=False)
        if not metrics_df.empty:
            metrics_df.to_excel(writer, sheet_name="metricas_modelos", index=False)
        if clip_log:
            pd.DataFrame(clip_log).to_excel(writer, sheet_name="diagnostico_zeros", index=False)
    return buf.getvalue()


def export_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sre_regressao_macro_{ts}.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_row(target, regiao, n_train, n_pred, metodo, r2, group_col="regiao"):
    return {
        "target": target,
        group_col: regiao,
        "n_train": n_train,
        "n_pred": n_pred,
        "metodo": metodo,
        "r2": r2,
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return (
        pd.DataFrame({"qtd_vazios": missing, "pct_vazios": pct})
        .query("qtd_vazios > 0")
        .sort_values("qtd_vazios", ascending=False)
    )


def train_pred_summary(
    df: pd.DataFrame,
    group_col: str,
    targets: list[str],
    selected_features: list[str],
    encoding_choices: dict[str, str],
) -> pd.DataFrame:
    """Return a summary table of n_train / n_pred per region × target."""
    rows = []
    for target in targets:
        for regiao in df[group_col].dropna().unique():
            mask_regiao = df[group_col] == regiao
            subset = df[mask_regiao]
            encoded, _ = encode_features(
                subset.reset_index(drop=False), selected_features, encoding_choices
            )
            encoded.index = subset.index
            has_feats = encoded.notna().all(axis=1)
            n_train = int((subset[target].notna() & has_feats).sum())
            n_pred = int((subset[target].isna() & has_feats).sum())
            rows.append(
                {
                    "target": target,
                    group_col: regiao,
                    "n_train": n_train,
                    "n_pred": n_pred,
                    "ok": "✅" if n_train >= 2 else ("⚠️ fallback" if n_pred > 0 else "—"),
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Equation formatters (plain text → LaTeX / Excel)
# ─────────────────────────────────────────────────────────────────────────────

def equation_to_latex(eq_str: str) -> str:
    """Convert a pipeline equation string to LaTeX format."""
    import re
    lines = eq_str.strip().split("\n")
    latex_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip Duan smearing notes
        if "Duan smearing" in line or "exp(log" in line:
            continue

        # Replace ŷ(name) with \\hat{y}_{\\text{name}}
        line = re.sub(r'ŷ\(([^)]+)\)', r'\\hat{y}_{\\text{\1}}', line)
        # Replace log(ŷ(name)) with \\log(\\hat{y}_{\\text{name}})
        line = re.sub(r'log\(ŷ\(([^)]+)\)\)', r'\\log(\\hat{y}_{\\text{\1}})', line)

        # Replace × with \times
        line = line.replace(" × ", " \\times ")

        # Escape underscores in feature names
        line = re.sub(r'(?<!\\)_', r'\\_', line)

        # Wrap in math mode
        latex_lines.append(f"${line}$")

    return "\n\\\\\n".join(latex_lines)


def equation_to_excel(eq_str: str) -> str:
    """Convert a pipeline equation string to an Excel formula (PT-BR format).
    Uses comma as decimal separator. Replace [feature] with cell references."""
    import re
    lines = eq_str.strip().split("\n")
    intercept = None
    terms = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Duan smearing" in line or "exp(log" in line:
            continue

        if " = " in line and intercept is None:
            rhs = line.split(" = ", 1)[1]
            match = re.match(r'([+-]?\s*[\d.]+)', rhs)
            if match:
                intercept = float(match.group(1).replace(" ", ""))
            continue
        
        match = re.match(r'([+-])\s+([\d.]+)\s+×\s+(.+)', line)
        if match:
            sign = match.group(1)
            coef = float(match.group(2))
            feature = match.group(3).strip()
            terms.append((sign, coef, feature))

    if intercept is None:
        return ""

    def ptbr(n):
        return f"{n:.4f}".replace(".", ",")

    formula = f"={ptbr(intercept)}"
    for sign, coef, feature in terms:
        formula += f" {sign} {ptbr(coef)}*{feature}"
    
    return formula
