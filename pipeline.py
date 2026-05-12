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
    """Return candidate feature columns split by type, excluding targets, group_col,
    constant columns (nunique == 1), and near-ID columns (nunique / nrows > 0.95)."""
    exclude = set(targets) | {group_col}
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
) -> tuple[list[int], list[str], object | None]:
    """
    Iteratively remove the feature with the highest p-value above `threshold`.
    Returns:
        kept_indices   - column indices in X_sc that survived
        kept_names     - corresponding feature names
        ols_result     - fitted statsmodels OLS result (or None if 0 features)
    """
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
        max_pval = pvals.max()

        if max_pval <= threshold:
            break

        # Remove the worst feature
        worst_local = int(pvals.argmax())
        kept.pop(worst_local)

    return kept, [feature_names[i] for i in kept], res


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_pipeline(
    df: pd.DataFrame,
    group_col: str,
    targets: list[str],
    selected_features: list[str],
    encoding_choices: dict[str, str],
    pvalue_threshold: float,
    progress_callback=None,  # callable(fraction: float, msg: str)
) -> dict:
    """
    Full prediction pipeline.

    Returns a dict with keys:
        df_result     - original df with predicted values filled in
        metrics_df    - one row per (target, region) with R², RMSE, RSE, …
        pvalues_df    - one row per (target, region, feature) with p-value
        coefs_df      - one row per (target, region, feature) with coefficient
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

    regions = df[group_col].dropna().unique()
    total_steps = len(targets) * len(regions)
    step = 0

    for target in targets:
        for regiao in regions:
            step += 1
            if progress_callback:
                progress_callback(
                    step / total_steps,
                    f"Target: {target} | Região: {regiao}",
                )

            mask_regiao = df_result[group_col] == regiao
            subset = df_result[mask_regiao].copy()

            # Rows with all features present
            encoded_all, encoders = encode_features(
                subset.reset_index(drop=False),
                selected_features,
                encoding_choices,
            )
            # Align index back to original
            encoded_all.index = subset.index

            feat_cols = list(encoded_all.columns)

            has_all_features = encoded_all.notna().all(axis=1)
            mask_train = subset[target].notna() & has_all_features
            mask_pred = subset[target].isna() & has_all_features

            n_train = int(mask_train.sum())
            n_pred = int(mask_pred.sum())

            if n_pred == 0:
                model_log.append(
                    _log_row(target, regiao, n_train, n_pred, "sem_vazios", None, group_col)
                )
                continue

            if n_train < 2:
                global_mean = df_result.loc[df_result[target].notna(), target].mean()
                idx_pred = subset[mask_pred].index
                df_result.loc[idx_pred, target] = global_mean
                fonte_cols[target].loc[idx_pred] = "previsto_media_global"
                model_log.append(
                    _log_row(target, regiao, n_train, n_pred, "media_global", None, group_col)
                )
                continue

            X_train_raw = encoded_all.loc[mask_train].values.astype(float)
            y_train = subset.loc[mask_train, target].values.astype(float)
            X_pred_raw = encoded_all.loc[mask_pred].values.astype(float)
            idx_pred = subset[mask_pred].index

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)
            X_pred_sc = scaler.transform(X_pred_raw)

            # R² do modelo completo (antes do backward elimination) — informativo
            try:
                _res_full = sm.OLS(y_train, sm.add_constant(X_train_sc, has_constant="add")).fit()
                r2_full = round(float(_res_full.rsquared), 4)
            except Exception:
                r2_full = None

            kept_idx, kept_names, ols_res = backward_elimination_ols(
                X_train_sc, y_train, feat_cols, pvalue_threshold
            )

            if not kept_idx or ols_res is None:
                # All features eliminated — fall back to global mean
                global_mean = df_result.loc[df_result[target].notna(), target].mean()
                df_result.loc[idx_pred, target] = global_mean
                fonte_cols[target].loc[idx_pred] = "previsto_media_global"
                model_log.append(
                    {
                        **_log_row(target, regiao, n_train, n_pred, "media_global_sem_features", None, group_col),
                        "r2_modelo_completo": r2_full,
                        "aviso": f"Todas as features eliminadas (p>{pvalue_threshold}). R²_completo={r2_full}. Aumente o threshold.",
                    }
                )
                continue

            X_train_final = X_train_sc[:, kept_idx]
            X_pred_final = X_pred_sc[:, kept_idx]

            # Predict
            X_pred_const = sm.add_constant(X_pred_final, has_constant="add")
            y_hat = ols_res.predict(X_pred_const)

            # Clip negatives
            n_neg = int((y_hat < 0).sum())
            if n_neg > 0:
                clip_log.append(
                    {
                        "target": target,
                        group_col: regiao,
                        "n_negativos_clampados": n_neg,
                        "min_previsto_bruto": round(float(y_hat.min()), 4),
                        "media_y_treino": round(float(y_train.mean()), 4),
                    }
                )
            y_hat = np.maximum(y_hat, 0)

            df_result.loc[idx_pred, target] = np.round(y_hat, 4)
            fonte_cols[target].loc[idx_pred] = "previsto"

            # Metrics
            r2 = float(ols_res.rsquared)
            X_train_ols = sm.add_constant(X_train_final, has_constant="add")
            y_pred_train = ols_res.predict(X_train_ols)
            residuals = y_train - y_pred_train
            n, p = X_train_final.shape
            denom = n - p - 1
            ss_res = float(np.sum(residuals ** 2))

            mse  = float(np.mean(residuals ** 2))
            mae  = float(np.mean(np.abs(residuals)))
            rmse = math.sqrt(mse)
            # MAPE: avoid division by zero
            nonzero = y_train != 0
            mape = float(np.mean(np.abs(residuals[nonzero] / y_train[nonzero])) * 100) if nonzero.any() else None
            rse  = math.sqrt(ss_res / denom) if denom > 0 else None

            model_log.append(
                {
                    **_log_row(target, regiao, n_train, n_pred, "regressao_linear", round(r2, 4), group_col),
                    "r2_modelo_completo": r2_full,
                    "mse":  round(mse,  4),
                    "mae":  round(mae,  4),
                    "rmse": round(rmse, 4),
                    "mape": round(mape, 4) if mape is not None else None,
                    "rse":  round(rse,  4) if rse  is not None else None,
                    "features_usadas": ", ".join(kept_names),
                    "n_features_usadas": len(kept_names),
                    "aviso": None,
                }
            )

            # Store residuals for analysis
            for _obs, _pred, _resid in zip(y_train, y_pred_train, residuals):
                residuals_rows.append({
                    "target": target,
                    group_col: regiao,
                    "y_obs":   round(float(_obs),   4),
                    "y_pred":  round(float(_pred),  4),
                    "residuo": round(float(_resid), 4),
                })

            # P-values and coefficients (index 0 = const)
            for j, fname in enumerate(kept_names):
                pvalues_rows.append(
                    {
                        "target": target,
                        group_col: regiao,
                        "feature": fname,
                        "coef": round(float(ols_res.params[j + 1]), 6),
                        "pvalue": round(float(ols_res.pvalues[j + 1]), 6),
                        "significativo": ols_res.pvalues[j + 1] <= pvalue_threshold,
                    }
                )
                coefs_rows.append(
                    {
                        "target": target,
                        group_col: regiao,
                        "feature": fname,
                        "coef": round(float(ols_res.params[j + 1]), 6),
                    }
                )
            # ── Build equation string ───────────────────────────────────────────────
            intercept = float(ols_res.params[0])
            padding = " " * (len(target) + 5)  # align continuation lines under the target
            eq_parts = [f"ŷ({target}) = {intercept:+.4f}"]
            for _j, _fname in enumerate(kept_names):
                _coef = float(ols_res.params[_j + 1])
                _sign = "+" if _coef >= 0 else "-"
                eq_parts.append(f"{padding}  {_sign} {abs(_coef):.4f} × {_fname}")
            equations_rows.append(
                {
                    "target": target,
                    group_col: regiao,
                    "intercept": round(intercept, 4),
                    "equation": "\n".join(eq_parts),
                    "n_features": len(kept_names),
                }
            )
    # ── Fonte columns ────────────────────────────────────────────────────────
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

    metrics_df    = pd.DataFrame(model_log)
    pvalues_df    = pd.DataFrame(pvalues_rows)   if pvalues_rows   else pd.DataFrame()
    coefs_df      = pd.DataFrame(coefs_rows)     if coefs_rows     else pd.DataFrame()
    equations_df  = pd.DataFrame(equations_rows) if equations_rows else pd.DataFrame()
    residuals_df  = pd.DataFrame(residuals_rows) if residuals_rows else pd.DataFrame()

    return {
        "df_result":    df_result,
        "metrics_df":   metrics_df,
        "pvalues_df":   pvalues_df,
        "coefs_df":     coefs_df,
        "equations_df": equations_df,
        "residuals_df": residuals_df,
        "clip_log":     clip_log,
        "fonte_cols":   fonte_cols,
        "group_col":    group_col,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(result: dict) -> bytes:
    """Serialize the pipeline result to an in-memory Excel workbook and return
    the raw bytes (ready to pass to st.download_button)."""
    import io

    df_result = result["df_result"].copy()
    metrics_df = result["metrics_df"]
    clip_log = result["clip_log"]
    targets = list(result.get("fonte_cols", {}).keys())

    # Drop per-target fonte_* columns — keep only the unified 'fonte' column
    fonte_cols_to_drop = [c for c in df_result.columns if c.startswith("fonte_")]
    df_result.drop(columns=fonte_cols_to_drop, inplace=True)

    # Simplify 'fonte': map anything predicted to 'previsto', keep 'observado'
    if "fonte" in df_result.columns:
        df_result["fonte"] = df_result["fonte"].apply(
            lambda v: "observado" if v == "observado" else "previsto"
        )

    # Drop booleanised helper columns (pattern: <original>_is_<values>)
    import re
    bool_pattern = re.compile(r".+_is_.+")
    bool_cols = [c for c in df_result.columns if bool_pattern.fullmatch(c)]
    df_result.drop(columns=bool_cols, inplace=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="dados", index=False)
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
