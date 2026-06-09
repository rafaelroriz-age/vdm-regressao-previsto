"""
charts.py
---------
All Plotly figure factories for the Streamlit app.
Each function returns a plotly.graph_objects.Figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _fmt_pivot(z: np.ndarray, fmt: str) -> np.ndarray:
    """Format a 2-D numeric array as strings, replacing NaN with empty string."""
    return np.vectorize(lambda v: ("" if np.isnan(v) else format(v, fmt)))(z)


# ─────────────────────────────────────────────────────────────────────────────
# R² Heatmap  (target × region)
# ─────────────────────────────────────────────────────────────────────────────

def plot_r2_heatmap(metrics_df: pd.DataFrame, group_col: str = "regiao") -> go.Figure:
    df = metrics_df[metrics_df["metodo"].isin(["regressao_linear", "regressao_linear_strat"])].copy()
    if df.empty:
        return _empty_fig("Sem dados de regressão linear para heatmap de R²")

    pivot = df.pivot_table(index="target", columns=group_col, values="r2", aggfunc="mean")
    pivot.columns = pivot.columns.astype(str)
    _z = pivot.values.astype(float)
    fig = px.imshow(
        pivot,
        text_auto=False,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        title="R² por Target × Região Macro",
        labels={"color": "R²"},
        aspect="auto",
    )
    fig.update_traces(text=_fmt_pivot(_z, ".3f"), texttemplate="%{text}")
    fig.update_xaxes(type="category")
    fig.update_layout(
        coloraxis_colorbar_title="R²",
        xaxis_title="Região Macro",
        yaxis_title="Target",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# R² Boxplot  (distribution across regions, per target)
# ─────────────────────────────────────────────────────────────────────────────

def plot_r2_boxplot(metrics_df: pd.DataFrame) -> go.Figure:
    df = metrics_df[metrics_df["metodo"].isin(["regressao_linear", "regressao_linear_strat"])].dropna(subset=["r2"])
    if df.empty:
        return _empty_fig("Sem dados de regressão linear para boxplot de R²")

    fig = px.box(
        df,
        x="target",
        y="r2",
        color="target",
        points="all",
        title="Distribuição de R² por Target (entre regiões)",
        labels={"r2": "R²", "target": "Target"},
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.05], margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RMSE Grouped Bar  (target × region)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_bar(metrics_df: pd.DataFrame, group_col: str = "regiao") -> go.Figure:
    df = metrics_df[metrics_df["metodo"].isin(["regressao_linear", "regressao_linear_strat"])].dropna(subset=["rmse"])
    if df.empty:
        return _empty_fig("Sem dados de regressão linear para gráfico de RMSE")

    fig = px.bar(
        df,
        x=group_col,
        y="rmse",
        color="target",
        barmode="group",
        title="RMSE por Região × Target",
        labels={"rmse": "RMSE", group_col: "Região Macro", "target": "Target"},
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Observed vs Predicted Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_obs_vs_pred(
    df_orig: pd.DataFrame,
    df_result: pd.DataFrame,
    target: str,
    group_col: str | None = None,
) -> go.Figure:
    """
    Scatter of true (observed) values vs. in-sample fitted values.
    When group_col is None, all points are shown in a single colour.
    """
    mask_obs = df_orig[target].notna()
    if mask_obs.sum() < 2:
        return _empty_fig(f"Poucos dados observados para {target}")

    df_plot = df_orig[mask_obs].copy()
    y_true = df_plot[target].values
    y_pred = df_result.loc[mask_obs, target].values

    df_scatter = pd.DataFrame({"Observado": y_true, "Ajustado": y_pred})
    color_col = None
    if group_col and group_col in df_plot.columns:
        df_scatter["Região"] = df_plot[group_col].astype(str).values
        color_col = "Região"

    fig = px.scatter(
        df_scatter,
        x="Observado",
        y="Ajustado",
        color=color_col,
        title=f"Observado vs Ajustado — {target}",
        opacity=0.7,
    )

    # 45° perfect-fit line
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    fig.add_shape(
        type="line",
        x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(color="gray", dash="dash", width=1.5),
    )
    fig.update_layout(margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient Heatmap  (feature × region, for a given target)
# ─────────────────────────────────────────────────────────────────────────────

def plot_coef_heatmap(
    coefs_df: pd.DataFrame, target: str, group_col: str = "regiao"
) -> go.Figure:
    df = coefs_df[coefs_df["target"] == target]
    if df.empty:
        return _empty_fig(f"Sem coeficientes disponíveis para {target}")

    pivot = df.pivot_table(index="feature", columns=group_col, values="coef", aggfunc="mean")
    pivot.columns = pivot.columns.astype(str)
    _z = pivot.values.astype(float)
    fig = px.imshow(
        pivot,
        text_auto=False,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"Coeficientes OLS — {target}",
        labels={"color": "Coef"},
        aspect="auto",
    )
    fig.update_traces(text=_fmt_pivot(_z, ".4f"), texttemplate="%{text}")
    fig.update_xaxes(type="category")
    fig.update_layout(
        xaxis_title="Região Macro",
        yaxis_title="Feature",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# P-value Heatmap  (feature × region, for a given target)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pvalue_heatmap(
    pvalues_df: pd.DataFrame,
    target: str,
    group_col: str = "regiao",
    threshold: float = 0.05,
) -> go.Figure:
    df = pvalues_df[pvalues_df["target"] == target]
    if df.empty:
        return _empty_fig(f"Sem p-values disponíveis para {target}")

    pivot = df.pivot_table(index="feature", columns=group_col, values="pvalue", aggfunc="mean")
    pivot.columns = pivot.columns.astype(str)

    # Custom discrete colour: green if ≤ threshold, red otherwise
    z = pivot.values.astype(float)
    text = np.vectorize(lambda v: f"{v:.4f}" if not np.isnan(v) else "")(z)

    fig = px.imshow(
        pivot,
        text_auto=False,
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=max(float(np.nanmax(z)), threshold * 2) if not np.all(np.isnan(z)) else 0.2,
        title=f"P-values por Feature × Região — {target}  (threshold={threshold})",
        labels={"color": "p-value"},
        aspect="auto",
    )

    # Overlay text
    fig.update_traces(text=text, texttemplate="%{text}")
    fig.update_xaxes(type="category")

    # Add threshold reference line annotation
    fig.add_annotation(
        text=f"verde ≤ {threshold} (significativo)",
        xref="paper", yref="paper",
        x=1.0, y=-0.12,
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="right",
    )
    fig.update_layout(
        xaxis_title="Região Macro",
        yaxis_title="Feature",
        margin=dict(l=10, r=10, t=60, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance bar (mean |coef| across regions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(coefs_df: pd.DataFrame, target: str) -> go.Figure:
    df = coefs_df[coefs_df["target"] == target]
    if df.empty:
        return _empty_fig(f"Sem coeficientes disponíveis para {target}")

    imp = (
        df.groupby("feature")["coef"]
        .apply(lambda s: s.abs().mean())
        .reset_index()
        .rename(columns={"coef": "importancia_media"})
        .sort_values("importancia_media", ascending=True)
    )

    fig = px.bar(
        imp,
        x="importancia_media",
        y="feature",
        orientation="h",
        title=f"Importância Média das Features (|coef| médio) — {target}",
        labels={"importancia_media": "|Coeficiente| médio", "feature": "Feature"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Residual histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals_hist(
    df_orig: pd.DataFrame, df_result: pd.DataFrame, target: str
) -> go.Figure:
    mask = df_orig[target].notna()
    if mask.sum() < 2:
        return _empty_fig(f"Poucos dados para resíduos de {target}")

    residuals = df_orig.loc[mask, target].values - df_result.loc[mask, target].values
    fig = px.histogram(
        x=residuals,
        nbins=30,
        title=f"Distribuição de Resíduos — {target}",
        labels={"x": "Resíduo (Observado − Ajustado)"},
        opacity=0.75,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(showlegend=False, margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric pivot tables (R², RMSE, MAE, MAPE by target × region)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_pivot(
    metrics_df: pd.DataFrame,
    metric: str,
    group_col: str,
    fmt: str = ".4f",
    title: str | None = None,
    colorscale: str = "RdYlGn",
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    df = metrics_df[metrics_df["metodo"].isin(["regressao_linear", "regressao_linear_strat"])].dropna(subset=[metric])
    if df.empty:
        return _empty_fig(f"Sem dados de '{metric}' para exibir")
    pivot = df.pivot_table(index="target", columns=group_col, values=metric, aggfunc="mean")
    pivot.columns = pivot.columns.astype(str)
    _z = pivot.values.astype(float)
    fig = px.imshow(
        pivot,
        text_auto=False,
        color_continuous_scale=colorscale,
        zmin=zmin,
        zmax=zmax,
        title=title or f"{metric.upper()} por Target × Região Macro",
        labels={"color": metric.upper()},
        aspect="auto",
    )
    fig.update_traces(text=_fmt_pivot(_z, fmt), texttemplate="%{text}")
    fig.update_xaxes(type="category")
    fig.update_layout(
        xaxis_title="Região Macro",
        yaxis_title="Target",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Residuals vs Fitted
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals_vs_fitted(
    residuals_df: pd.DataFrame, target: str, group_col: str | None = None
) -> go.Figure:
    df = residuals_df[residuals_df["target"] == target]
    if df.empty:
        return _empty_fig(f"Sem resíduos para {target}")
    fig = px.scatter(
        df,
        x="y_pred",
        y="residuo",
        color=group_col if group_col else None,
        opacity=0.65,
        title=f"Resíduos vs Valores Ajustados — {target}",
        labels={"y_pred": "Valor Ajustado (ŷ)", "residuo": "Resíduo (y − ŷ)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Q-Q Normal Plot of Residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_qq(residuals_df: pd.DataFrame, target: str) -> go.Figure:
    import scipy.stats as stats

    df = residuals_df[residuals_df["target"] == target]
    if df.empty or len(df) < 4:
        return _empty_fig(f"Poucos dados para Q-Q de {target}")

    res = df["residuo"].values
    (osm, osr), (slope, intercept, _) = stats.probplot(res, dist="norm")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr,
        mode="markers",
        name="Resíduos",
        marker=dict(size=5, opacity=0.7),
    ))
    # Reference line
    x_line = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="Linha Normal",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(
        title=f"Q-Q Normal dos Resíduos — {target}",
        xaxis_title="Quantis Teóricos (Normal)",
        yaxis_title="Quantis Observados",
        margin=dict(t=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Scale-Location (√|residuals| vs fitted)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scale_location(
    residuals_df: pd.DataFrame, target: str, group_col: str | None = None
) -> go.Figure:
    df = residuals_df[residuals_df["target"] == target].copy()
    if df.empty:
        return _empty_fig(f"Sem resíduos para {target}")
    df["sqrt_abs_res"] = np.sqrt(np.abs(df["residuo"]))
    fig = px.scatter(
        df,
        x="y_pred",
        y="sqrt_abs_res",
        color=group_col if group_col else None,
        opacity=0.65,
        title=f"Scale-Location (√|resíduo|) — {target}",
        labels={
            "y_pred": "Valor Ajustado (ŷ)",
            "sqrt_abs_res": "√|Resíduo|",
        },
    )
    fig.update_layout(margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Residuals histogram from residuals_df
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals_hist_df(
    residuals_df: pd.DataFrame, target: str, group_col: str | None = None
) -> go.Figure:
    df = residuals_df[residuals_df["target"] == target]
    if df.empty:
        return _empty_fig(f"Sem resíduos para {target}")
    fig = px.histogram(
        df,
        x="residuo",
        color=group_col if group_col else None,
        nbins=30,
        barmode="overlay" if group_col else "relative",
        opacity=0.7,
        title=f"Distribuição dos Resíduos — {target}",
        labels={"residuo": "Resíduo (y − ŷ)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(margin=dict(t=50))
    return fig


# ─────────────────────────────────────────────────────────────────────────────# Global R² Bar (one bar per target, using the global model R²)
# ──────────────────────────────────────────────────────────────────────────────

def plot_global_r2_bar(metrics_df: pd.DataFrame) -> go.Figure:
    df = metrics_df[metrics_df["metodo"].isin(["regressao_linear", "regressao_linear_strat"])].copy()
    if df.empty:
        return _empty_fig("Sem dados de regressão linear")
    # r2_global is the same for all rows of a given target — take first non-null
    if "r2_global" in df.columns:
        global_r2 = (
            df.dropna(subset=["r2_global"])
            .groupby("target")["r2_global"]
            .first()
            .reset_index()
            .rename(columns={"r2_global": "R²"})
        )
    else:
        global_r2 = (
            df.groupby("target")["r2"]
            .mean()
            .reset_index()
            .rename(columns={"r2": "R²"})
        )
    fig = px.bar(
        global_r2,
        x="target",
        y="R²",
        color="target",
        text_auto=".4f",
        title="R² Global por Target",
        labels={"target": "Target", "R²": "R² (modelo global)"},
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(showlegend=False, margin=dict(t=50, b=20))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Global Metric Bars (RMSE, MAE, MAPE computed from raw residuals)
# ──────────────────────────────────────────────────────────────────────────────

def plot_global_metric_bars(residuals_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of RMSE, MAE and MAPE per target (all regions combined)."""
    if residuals_df.empty:
        return _empty_fig("Sem resíduos para métricas globais")
    rows = []
    for t, g in residuals_df.groupby("target"):
        resid = g["residuo"].values
        y_obs = g["y_obs"].values
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        nonzero = y_obs != 0
        mape = float(np.mean(np.abs(resid[nonzero] / y_obs[nonzero])) * 100) if nonzero.any() else 0.0
        rows.append({"Target": str(t), "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})
    if not rows:
        return _empty_fig("Sem dados para métricas globais")
    df = pd.DataFrame(rows)
    df_m = df.melt("Target", var_name="Métrica", value_name="Valor")
    fig = px.bar(
        df_m,
        x="Target",
        y="Valor",
        color="Métrica",
        barmode="group",
        text_auto=".2f",
        title="Métricas Globais por Target (RMSE, MAE, MAPE)",
        labels={"Valor": "Valor", "Target": "Target"},
    )
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


# ──────────────────────────────────────────────────────────────────────────────# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(t=30, b=10),
    )
    return fig
