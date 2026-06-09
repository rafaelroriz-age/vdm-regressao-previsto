"""
app.py
------
Streamlit front-end for the VDM regression pipeline.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import pipeline as pl

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VDM Predict — Regressão por Região Macro",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "df": None,
    "df_filename": "",
    "result": None,
    "ran": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuração")

    # ── 1. Upload ────────────────────────────────────────────────────────────
    st.subheader("1. Dataset")
    uploaded = st.file_uploader(
        "Arraste ou selecione o arquivo",
        type=["xlsx", "xls", "csv"],
        help="Formatos suportados: .xlsx, .xls, .csv",
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded)
            else:
                df_new = pd.read_excel(uploaded)

            if (
                st.session_state["df"] is None
                or uploaded.name != st.session_state["df_filename"]
            ):
                st.session_state["df"] = df_new
                st.session_state["df_filename"] = uploaded.name
                st.session_state["result"] = None
                st.session_state["ran"] = False
                st.success(f"✅ Carregado: {uploaded.name} ({df_new.shape[0]:,} linhas × {df_new.shape[1]} colunas)")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    df: pd.DataFrame | None = st.session_state["df"]

    if df is None:
        st.info("Faça upload de um arquivo para continuar.")
        st.stop()

    st.divider()

    # ── 2. Coluna agrupadora ─────────────────────────────────────────────────
    st.subheader("2. Coluna Agrupadora")
    group_default = "regional_macro" if "regional_macro" in df.columns else df.columns[0]
    group_col = st.selectbox(
        "Coluna de grupo (região macro)",
        options=list(df.columns),
        index=list(df.columns).index(group_default) if group_default in df.columns else 0,
    )

    st.divider()

    # ── 2b. Fundir Grupos ────────────────────────────────────────────────────
    st.subheader("2b. Fundir Grupos")
    st.caption("Trate dois ou mais valores do grupo como um único grupo na regressão.")
    _group_values = sorted(df[group_col].dropna().map(pl.normalize_group_label).unique().tolist(), key=str)
    _n_merges = st.number_input(
        "Número de junções", min_value=0, max_value=5, value=1, step=1,
        help="Quantas junções de grupos definir.",
    )
    group_merge_defs: list[tuple[list, str]] = []
    for _mi in range(int(_n_merges)):
        _pre = [v for v in ["1", "2"] if v in _group_values] if _mi == 0 else []
        _sel = st.multiselect(
            f"Grupos a fundir #{_mi + 1}",
            options=_group_values,
            default=_pre,
            key=f"gmerge_{_mi}",
        )
        if len(_sel) >= 2:
            _lbl = "+".join(str(v) for v in _sel)
            group_merge_defs.append((_sel, _lbl))
            st.caption(f"→ {_sel} serão tratados como **`{_lbl}`**")

    # df com grupos fundidos — base para todas as etapas seguintes
    df_base = pl.apply_group_merge(df, group_col, group_merge_defs)

    st.divider()

    # ── 3. Targets ───────────────────────────────────────────────────────────
    st.subheader("3. Colunas Target")
    default_targets = [c for c in ["vmd", "vmdc", "n_aashto", "n_usace"] if c in df.columns]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    targets = st.multiselect(
        "Colunas a prever (com vazios)",
        options=numeric_cols,
        default=default_targets if default_targets else numeric_cols[:1],
    )

    st.divider()

    # ── 4. Transformações (boolean merge) ─────────────────────────────────────
    st.subheader("4. Transformações")
    st.caption("Booleanize colunas categóricas de baixa cardinalidade antes de usar como feature.")
    bool_merges: dict[str, tuple[list, str]] = {}
    _init_candidates = pl.split_candidates(df_base, targets, group_col)
    for _col in _init_candidates["categorical"]:
        if df_base[_col].nunique() <= 6:
            _unique = sorted(df_base[_col].dropna().astype(str).unique().tolist())
            _default_true = ["DUP"] if _col == "situacao" else []
            with st.expander(
                f"`{_col}` — ({' / '.join(_unique)})",
                expanded=(_col == "situacao"),
            ):
                _pos = st.multiselect(
                    "Valores = **True** (demais = False)",
                    options=_unique,
                    default=[v for v in _default_true if v in _unique],
                    key=f"boolmerge_{_col}",
                    help="As categorias selecionadas viram True; as demais viram False.",
                )
                if _pos:
                    _new_name = f"{_col}_is_{'_'.join(v.lower() for v in _pos)}"
                    bool_merges[_col] = (_pos, _new_name)
                    st.caption(f"→ nova coluna booleana: **`{_new_name}`**")

    # df_proc é usado em todas as etapas abaixo
    df_proc = pl.apply_boolean_merge(df_base, bool_merges) if bool_merges else df_base
    df_proc = pl.feature_engineering(df_proc, targets, group_col)

    st.divider()

    # ── 5. Feature selection ──────────────────────────────────────────────────
    st.subheader("5. Features por Target")
    st.caption("Configure features e obrigatórias independentemente para cada target.")
    candidates_shared = pl.split_candidates(df_proc, targets, group_col)

    # Global encoding choices (same for all targets)
    encoding_choices: dict[str, str] = {}
    if candidates_shared["categorical"]:
        with st.expander("Encoding por coluna categórica", expanded=True):
            for col in candidates_shared["categorical"]:
                n_unique = df_proc[col].nunique()
                encoding_choices[col] = st.radio(
                    f"`{col}` ({n_unique} categorias)" + (" ⭐ coluna de região" if col == group_col else ""),
                    options=["onehot", "label"],
                    format_func=lambda x: "One-Hot (colunas binárias)" if x == "onehot" else "Label (ordinal numérico)",
                    horizontal=True,
                    key=f"enc_{col}",
                )

    features_per_target: dict[str, list[str]] = {}
    mandatory_per_target: dict[str, list[str]] = {}
    if not targets:
        st.info("Selecione pelo menos 1 target acima.")
    else:
        for t in targets:
            with st.expander(f"🎯 **{t}**", expanded=False):
                # Candidates: own target included, other targets excluded
                other_targets = [ot for ot in targets if ot != t]
                cands_t = pl.split_candidates(df_proc, other_targets, group_col)
                all_feats_t = cands_t["numeric"] + cands_t["boolean"] + cands_t["categorical"]
                feats_t = st.multiselect(
                    "Features",
                    options=all_feats_t,
                    default=[f for f in all_feats_t if f != t],
                    key=f"feats_{t}",
                    help=f"'{t}' pode ser selecionado como feature — apenas seus valores já observados no dataset serão usados no treino.",
                )
                mandatory_t = st.multiselect(
                    "Obrigatórias (nunca removidas pelo backward elimination)",
                    options=feats_t,
                    default=[],
                    key=f"mandatory_{t}",
                    help="Features mantidas no modelo independente do p-value.",
                )
                if mandatory_t:
                    st.caption(f"✅ {len(mandatory_t)} obrigatória(s): {', '.join(mandatory_t)}")
                features_per_target[t] = feats_t
                mandatory_per_target[t] = mandatory_t

    # Union of all selected features (used in tab_config preview and run button check)
    selected_features = list(dict.fromkeys(f for feats in features_per_target.values() for f in feats))

    st.divider()

    # ── 6. P-value threshold ──────────────────────────────────────────────────
    st.subheader("6. P-value Threshold")
    pvalue_threshold = st.slider(
        "Backward elimination — remover features com p > threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.2f",
    )
    st.caption(
        f"Features com p-value > **{pvalue_threshold}** são removidas iterativamente antes de cada predição."
    )

    st.divider()
    # ── 6c. Log-transform ──────────────────────────────────────────
    st.subheader("6c. Log-transform")
    use_log_transform = st.checkbox(
        "Aplicar log-transform nos targets",
        value=True,
        help=(
            "Regride log(target) e aplica exp(ŷ) após → predições sempre positivas. "
            "Recomendado para VDM, N_AASHTO, N_USACE (distribuição lognormal)."
        ),
    )
    if use_log_transform:
        st.caption("✅ Predições sempre > 0. R² reportado no espaço log; RMSE/MAE no espaço original.")
    else:
        st.caption("⚠️ Sem log-transform: predições negativas serão clipadas para 0.")

    st.divider()
    # ── 6d. Limite de features ─────────────────────────────────────
    st.subheader("6d. Limite de Features")
    _max_feat_total = max(1, len(selected_features)) if selected_features else 20
    use_max_features = st.checkbox(
        "Limitar número máximo de features",
        value=False,
        help="Após o backward elimination, mantém apenas as N features com maior |t-stat|.",
    )
    max_features: int | None = None
    if use_max_features:
        max_features = st.slider(
            "Máximo de features no modelo",
            min_value=1,
            max_value=max(1, _max_feat_total),
            value=min(5, _max_feat_total),
            step=1,
        )
        st.caption(
            f"O modelo usará no máximo **{max_features}** feature(s). "
            "Features obrigatórias são sempre preservadas."
        )

    st.divider()
    # ── 6e. Modelo estratificado por região ──────────────────────────────
    st.subheader("6e. Modelo Estratificado por Região")
    use_stratified = st.checkbox(
        "Treinar um OLS independente por região",
        value=False,
        help=(
            "Quando ativado, cada região recebe seu próprio modelo OLS com "
            "coeficientes independentes. Regiões com poucas observações "
            "(< limite abaixo) usam fallback para média regional. "
            "Isso captura heterogeneidade regional que o modelo global ignora."
        ),
    )
    min_train_region: int = 10
    if use_stratified:
        min_train_region = st.slider(
            "Mínimo de observações de treino por região",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Regiões com menos observações que este valor usarão fallback "
                 "(média regional) em vez de modelo OLS próprio.",
        )
        reg_obs = df_proc[group_col].value_counts().to_dict()
        reg_info = ", ".join(
            f"{k}: {int(v)} SREs" for k, v in sorted(reg_obs.items(), key=lambda x: str(x[0]))
        )
        st.caption(f"Distribuição de SREs por região: {reg_info}")
        st.caption(
            f"Com limite = {min_train_region}, "
            f"regiões com ≥ {min_train_region} observações de treino "
            f"recebem OLS próprio; demais usam média regional."
        )

    st.divider()
    # ── 6f. Clipagem de predições ────────────────────────────────────────
    st.subheader("6f. Clipagem de Predições")
    use_clip = st.checkbox(
        "Limitar predições ao intervalo [0,3×mediana, 3,0×mediana] por região",
        value=False,
        help=(
            "Após a predição, valores previstos são clipados ao intervalo "
            "[30% da mediana regional, 300% da mediana regional]. "
            "Isso impede predições extremamente fora da escala típica da região, "
            "reduzindo erros percentuais aberrantes (> ±80%). "
            "Valores observados nunca são alterados."
        ),
    )
    if use_clip:
        st.caption("✅ Predições limitadas a [0,3×mediana, 3,0×mediana] por região.")
        st.caption("📋 Na exportação, colunas *_confianca marcam predições fora de [P10, P90].")

    st.divider()
    # ── 6. Run ───────────────────────────────────────────────────────────────
    run_disabled = len(targets) == 0 or not selected_features
    run_btn = st.button(
        "▶ Executar Pipeline",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    )
    if run_disabled:
        st.caption("⚠️ Selecione pelo menos 1 target e 1 feature.")

    # ── 7. Export ─────────────────────────────────────────────────────────────
    if st.session_state["ran"] and st.session_state["result"] is not None:
        excel_bytes = pl.export_to_excel(st.session_state["result"])
        st.download_button(
            label="⬇️ Exportar Excel",
            data=excel_bytes,
            file_name=pl.export_filename(),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    progress_bar = st.progress(0, text="Iniciando pipeline…")

    def _progress(fraction: float, msg: str):
        progress_bar.progress(fraction, text=msg)

    try:
        if use_stratified:
            result = pl.run_regression_pipeline_stratified(
                df=df_proc,
                group_col=group_col,
                targets=targets,
                features_per_target=features_per_target,
                encoding_choices=encoding_choices,
                pvalue_threshold=pvalue_threshold,
                log_transform=use_log_transform,
                mandatory_per_target=mandatory_per_target,
                max_features=max_features,
                min_train_region=min_train_region,
                clip_predictions=use_clip,
                progress_callback=_progress,
            )
        else:
            result = pl.run_regression_pipeline(
                df=df_proc,
                group_col=group_col,
                targets=targets,
                features_per_target=features_per_target,
                encoding_choices=encoding_choices,
                pvalue_threshold=pvalue_threshold,
                log_transform=use_log_transform,
                mandatory_per_target=mandatory_per_target,
                max_features=max_features,
                clip_predictions=use_clip,
                progress_callback=_progress,
            )
        st.session_state["result"] = result
        st.session_state["ran"] = True
        
        # If stratified, also run global model for global equations
        if use_stratified:
            try:
                result_global = pl.run_regression_pipeline(
                    df=df_proc,
                    group_col=group_col,
                    targets=targets,
                    features_per_target=features_per_target,
                    encoding_choices=encoding_choices,
                    pvalue_threshold=pvalue_threshold,
                    log_transform=use_log_transform,
                    mandatory_per_target=mandatory_per_target,
                    max_features=max_features,
                    clip_predictions=use_clip,
                )
                result["equations_global"] = result_global.get("equations_df", pd.DataFrame())
            except Exception:
                result["equations_global"] = pd.DataFrame()
        else:
            # Already global — equations_df IS the global equation
            result["equations_global"] = result.get("equations_df", pd.DataFrame())
        
        progress_bar.progress(1.0, text="Pipeline concluída! ✅")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Erro durante o pipeline: {e}")
        st.stop()

result = st.session_state.get("result")

# ─────────────────────────────────────────────────────────────────────────────
# Main content — 5 tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_dados, tab_config, tab_resultados, tab_metricas, tab_graficos = st.tabs(
    ["📋 Dados", "🔧 Configuração", "📊 Resultados", "📈 Métricas", "📉 Gráficos"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DADOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_dados:
    st.header("Visão Geral do Dataset")

    type_map = pl.detect_column_types(df_proc)
    n_num = sum(1 for v in type_map.values() if v == "numeric")
    n_bool = sum(1 for v in type_map.values() if v == "boolean")
    n_cat = sum(1 for v in type_map.values() if v == "categorical")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{df.shape[0]:,}")
    c2.metric("Colunas", f"{df.shape[1]}")
    c3.metric("Colunas numéricas", n_num)
    c4.metric("Colunas categóricas", n_cat + n_bool)

    st.subheader("Colunas com Valores Faltantes")
    missing_df = pl.missing_summary(df)
    if missing_df.empty:
        st.success("Nenhum valor faltante no dataset.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Preview do Dataset (primeiras 200 linhas)")
    st.dataframe(df.head(200), use_container_width=True, height=350)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

with tab_config:
    st.header("Configuração Atual")

    if not selected_features:
        st.warning("Nenhuma feature selecionada na sidebar.")
    else:
        # Feature list with type badges
        st.subheader("Features selecionadas")
        badge_color = {"numeric": "🔢", "boolean": "☑️", "categorical": "🏷️"}
        rows_feat = []
        for f in selected_features:
            t = type_map.get(f, "numeric")
            enc = encoding_choices.get(f, "—")
            n_miss = int(df_proc[f].isna().sum())
            rows_feat.append(
                {
                    "Feature": f,
                    "Tipo": f"{badge_color.get(t, '')} {t}",
                    "Encoding": enc if t == "categorical" else "—",
                    "Vazios": n_miss,
                    "% Vazio": f"{n_miss / len(df_proc) * 100:.1f}%",
                }
            )

        # Show count after one-hot expansion
        encoded_preview, _ = pl.encode_features(df_proc.head(5), selected_features, encoding_choices)
        n_expanded = len(encoded_preview.columns)
        st.info(
            f"**{len(selected_features)}** features selecionadas → "
            f"**{n_expanded}** colunas após encoding (one-hot expande categóricas em binárias)"
        )
        st.caption("ℹ️ Regressão global por SRE: um único modelo por target é treinado com todos os SREs. "
                   "Métricas por região (R², RMSE, etc.) são calculadas a partir dos resíduos do modelo global em cada região.")
        st.dataframe(pd.DataFrame(rows_feat), use_container_width=True)

    st.divider()
    st.subheader("Treino vs Predição por Região × Target")
    if targets and selected_features:
        with st.spinner("Calculando distribuição de treino/predição…"):
            summary_df = pl.train_pred_summary(df_proc, group_col, targets, selected_features, encoding_choices)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Selecione targets e features para ver o resumo.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_resultados:
    st.header("Resultados da Pipeline")

    if not st.session_state["ran"] or result is None:
        st.info("Execute a pipeline clicando em **▶ Executar Pipeline** na sidebar.")
    else:
        df_result: pd.DataFrame = result["df_result"]
        clip_log: list = result["clip_log"]
        group_col_res: str = result["group_col"]

        # Source distribution
        st.subheader("Distribuição da coluna 'fonte'")
        fonte_counts = df_result["fonte"].value_counts().reset_index()
        fonte_counts.columns = ["fonte", "qtd"]
        st.dataframe(fonte_counts, use_container_width=True)

        # Remaining nulls
        st.subheader("Vazios remanescentes após predição")
        remaining = df_result[targets].isnull().sum()
        if remaining.sum() == 0:
            st.success("✅ Nenhum vazio remanescente nos targets.")
        else:
            st.warning(f"⚠️ {remaining.sum()} vazio(s) ainda presentes (linhas sem features).")
            st.dataframe(remaining.to_frame("vazios_restantes"), use_container_width=True)

        # Preview predicted rows
        st.subheader("Preview das linhas previstas (primeiras 100)")
        mask_prev = df_result["fonte"].isin(["previsto", "misto", "previsto_media_global"])
        cols_show = [group_col_res] + selected_features + targets + ["fonte"]
        # Remove duplicates (group_col may already be in selected_features) keeping order
        _seen: set = set()
        cols_show = [c for c in cols_show if c in df_result.columns and not (c in _seen or _seen.add(c))]
        predicted_rows = df_result.loc[mask_prev, cols_show]
        st.dataframe(predicted_rows.head(100), use_container_width=True, height=350)
        st.caption(f"Total de linhas previstas: {mask_prev.sum():,}")

        # Clip / zero warning
        if clip_log:
            st.subheader("⚠️ Predições negativas zeradas")
            st.dataframe(pd.DataFrame(clip_log), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

with tab_metricas:
    st.header("Métricas dos Modelos")

    if not st.session_state["ran"] or result is None:
        st.info("Execute a pipeline para ver as métricas.")
    else:
        metrics_df: pd.DataFrame = result["metrics_df"]
        pvalues_df: pd.DataFrame = result["pvalues_df"]
        _gcol = result["group_col"]
        rl = metrics_df[metrics_df["metodo"] == "regressao_linear"]

        # ── Avisos de fallback ────────────────────────────────────────────
        _fallbacks = metrics_df[
            metrics_df["metodo"].isin(["media_global_sem_features", "media_global"])
            & metrics_df.get("aviso", pd.Series(dtype=str)).notna()
        ] if "aviso" in metrics_df.columns else pd.DataFrame()
        if not _fallbacks.empty:
            with st.expander(
                f"⚠️ {len(_fallbacks)} combinação(ões) em fallback — métricas None (clique para detalhes)",
                expanded=True,
            ):
                st.caption(
                    "Nestas combinações o backward elimination removeu **todas** as features. "
                    "A predição usou a média global. Para obter métricas, **aumente o threshold de P-value** na sidebar."
                )
                st.dataframe(
                    _fallbacks[[c for c in ["target", _gcol, "metodo", "r2_modelo_completo", "aviso"] if c in _fallbacks.columns]],
                    use_container_width=True,
                )

        st.divider()

        # ── R² pivot: uma linha por target, uma coluna por região ─────────
        st.subheader("R² por Target × Região")
        if not rl.empty:
            r2_pivot = (
                rl.pivot_table(index="target", columns=_gcol, values="r2", aggfunc="mean")
                .round(4)
            )
            # Append summary columns
            r2_pivot["Média"] = r2_pivot.mean(axis=1).round(4)
            r2_pivot["Mín"]   = r2_pivot.drop(columns=["Média"]).min(axis=1).round(4)
            r2_pivot["Máx"]   = r2_pivot.drop(columns=["Média", "Mín"]).max(axis=1).round(4)
            st.dataframe(
                r2_pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1,
                    subset=[c for c in r2_pivot.columns if c not in ("Média","Mín","Máx")]),
                use_container_width=True,
            )
        else:
            st.info("Nenhum modelo de regressão linear disponível.")

        st.divider()

        # ── Tabela completa de métricas ────────────────────────────────────
        st.subheader("Métricas Completas por Target × Região")
        metric_cols_ordered = [
            "target", _gcol, "n_train", "n_pred", "metodo",
            "r2", "r2_modelo_completo", "mse", "mae", "rmse", "mape", "rse",
            "n_features_usadas", "features_usadas", "aviso",
        ]
        metric_cols_present = [c for c in metric_cols_ordered if c in metrics_df.columns]
        st.dataframe(
            metrics_df[metric_cols_present].sort_values(["target", _gcol])
            if _gcol in metrics_df.columns else metrics_df[metric_cols_present],
            use_container_width=True,
            height=400,
        )

        # ── Pivot compacto para cada métrica ──────────────────────────────
        if not rl.empty:
            st.divider()
            st.subheader("Pivôs por Métrica")
            _metric_tabs = st.tabs(["MSE", "MAE", "RMSE", "MAPE", "RSE"])
            _metric_map = [
                ("mse",  "MSE",  "RdYlGn_r", None, None, ".2f"),
                ("mae",  "MAE",  "RdYlGn_r", None, None, ".2f"),
                ("rmse", "RMSE", "RdYlGn_r", None, None, ".2f"),
                ("mape", "MAPE (%)", "RdYlGn_r", None, None, ".2f"),
                ("rse",  "RSE",  "RdYlGn_r", None, None, ".4f"),
            ]
            for _tab, (_col, _label, _cs, _zmin, _zmax, _fmt) in zip(_metric_tabs, _metric_map):
                with _tab:
                    if _col in rl.columns and rl[_col].notna().any():
                        _piv = (
                            rl.pivot_table(index="target", columns=_gcol, values=_col, aggfunc="mean")
                            .round(4)
                        )
                        st.dataframe(
                            _piv.style.background_gradient(cmap="RdYlGn_r"),
                            use_container_width=True,
                        )
                    else:
                        st.info(f"Sem dados de {_label}.")

        st.divider()
        st.subheader("P-values por Feature × Região")
        if pvalues_df.empty:
            st.info("Nenhum p-value disponível (verifique se há regiões com treino ≥ 2).")
        else:
            # Filter controls
            p_target_filter = st.selectbox(
                "Filtrar por target",
                options=["Todos"] + list(pvalues_df["target"].unique()),
                key="pval_target_filter",
            )
            show_only_sig = st.checkbox(
                f"Mostrar apenas significativos (p ≤ {pvalue_threshold})",
                key="pval_sig_filter",
            )

            pv_view = pvalues_df.copy()
            if p_target_filter != "Todos":
                pv_view = pv_view[pv_view["target"] == p_target_filter]
            if show_only_sig:
                pv_view = pv_view[pv_view["pvalue"] <= pvalue_threshold]

            st.dataframe(
                pv_view.sort_values(["target", "pvalue"]),
                use_container_width=True,
                height=400,
            )

        st.divider()
        st.subheader("📐 Equações dos Modelos")
        equations_df = result.get("equations_df", pd.DataFrame())
        equations_global = result.get("equations_global", pd.DataFrame())
        
        if equations_df.empty and equations_global.empty:
            st.info("Nenhuma equação disponível (apenas fallbacks de média).")
        else:
            _eq_gcol = result["group_col"]
            
            eq_c1, eq_c2 = st.columns(2)
            with eq_c1:
                eq_target_sel = st.selectbox("Target", options=targets, key="eq_target_sel")
            with eq_c2:
                _eq_regions_list = sorted(equations_df[_eq_gcol].unique().tolist(), key=str) if not equations_df.empty else []
                _eq_region_opts = ["Global"] + _eq_regions_list
                eq_region_sel = st.selectbox("Região", options=_eq_region_opts, key="eq_region_sel")
            
            # Get the right equation
            _eq_row = None
            _is_global = (eq_region_sel == "Global")
            
            if _is_global and not equations_global.empty:
                _eq_row = equations_global[
                    (equations_global["target"] == eq_target_sel)
                ]
            elif not _is_global and not equations_df.empty:
                _eq_row = equations_df[
                    (equations_df["target"] == eq_target_sel)
                    & (equations_df[_eq_gcol] == eq_region_sel)
                ]
            
            if _eq_row is None or _eq_row.empty:
                st.info("Sem equação para esta combinação.")
            else:
                _r = _eq_row.iloc[0]
                eq_text = _r["equation"]
                
                # Generate formats
                eq_latex = pl.equation_to_latex(eq_text)
                eq_excel = pl.equation_to_excel(eq_text)
                
                # Display in 3 tabs
                eq_tab1, eq_tab2, eq_tab3 = st.tabs(["📝 Texto", "📐 LaTeX", "📊 Fórmula Excel"])
                
                with eq_tab1:
                    st.code(eq_text, language=None)
                
                with eq_tab2:
                    if eq_latex:
                        st.code(eq_latex, language="latex")
                        st.caption("Copie e cole em qualquer editor LaTeX.")
                    else:
                        st.info("Formato LaTeX indisponível para esta equação.")
                
                with eq_tab3:
                    if eq_excel:
                        st.code(eq_excel, language=None)
                        st.caption(
                            "Fórmula em português (vírgula como decimal). "
                            "Substitua cada `feature` pela célula correspondente "
                            "(ex: `classe_Longitudinais` → `D2`). "
                            "Se os dados estiverem como Tabela (Ctrl+T), use "
                            "`=[@[classe_Longitudinais]]`."
                        )
                    else:
                        st.info("Fórmula Excel indisponível para esta equação.")
                
                st.caption(
                    f"Intercepto: **{_r['intercept']:+.4f}**  |  "
                    f"Features: **{_r['n_features']}**"
                    + (f"  |  Smearing: **{_r['smearing']:.6f}**" if pd.notna(_r.get('smearing')) else "")
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_graficos:
    st.header("Gráficos")

    if not st.session_state["ran"] or result is None:
        st.info("Execute a pipeline para ver os gráficos.")
    else:
        metrics_df = result["metrics_df"]
        pvalues_df = result["pvalues_df"]
        coefs_df = result["coefs_df"]
        residuals_df = result.get("residuals_df", pd.DataFrame())
        df_result = result["df_result"]
        group_col_res = result["group_col"]

        # ── Row 1: R² Global por Target ──────────────────────────────────
        st.plotly_chart(
            ch.plot_global_r2_bar(metrics_df),
            use_container_width=True,
        )

        st.divider()

        # ── Row 2: Métricas Globais (RMSE, MAE, MAPE) ────────────────────
        st.plotly_chart(
            ch.plot_global_metric_bars(residuals_df),
            use_container_width=True,
        )

        st.divider()

        # ── Row 3: Obs vs Pred scatter ────────────────────────────────────
        st.subheader("Observado vs Ajustado")
        chart_target = st.selectbox(
            "Target",
            options=targets,
            key="chart_target",
        )
        st.plotly_chart(
            ch.plot_obs_vs_pred(df_proc, df_result, chart_target),
            use_container_width=True,
        )

        st.divider()

        # ── Row 4: Análise de Resíduos ────────────────────────────────────
        st.subheader("Análise de Resíduos")
        res_target = st.selectbox(
            "Target para análise de resíduos",
            options=targets,
            key="res_target",
        )
        if residuals_df.empty:
            st.info("Sem resíduos disponíveis (nenhuma regressão linear foi executada).")
        else:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.plotly_chart(
                    ch.plot_residuals_vs_fitted(residuals_df, res_target),
                    use_container_width=True,
                )
            with rc2:
                st.plotly_chart(
                    ch.plot_residuals_hist_df(residuals_df, res_target),
                    use_container_width=True,
                )
            rc3, rc4 = st.columns(2)
            with rc3:
                st.plotly_chart(
                    ch.plot_qq(residuals_df, res_target),
                    use_container_width=True,
                )
            with rc4:
                st.plotly_chart(
                    ch.plot_scale_location(residuals_df, res_target),
                    use_container_width=True,
                )

        st.divider()

        # ── Row 5: Coef heatmap + Feature importance ──────────────────────
        if not coefs_df.empty:
            st.subheader("Coeficientes e Importância das Features")
            coef_target = st.selectbox(
                "Target para coeficientes",
                options=targets,
                key="coef_target",
            )
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(
                    ch.plot_coef_heatmap(coefs_df, coef_target, group_col=group_col_res),
                    use_container_width=True,
                )
            with col6:
                st.plotly_chart(
                    ch.plot_feature_importance(coefs_df, coef_target),
                    use_container_width=True,
                )

        st.divider()

        # ── Row 6: P-value heatmap ────────────────────────────────────────
        if not pvalues_df.empty:
            st.subheader("Heatmap de P-values")
            pv_target = st.selectbox(
                "Target para p-values",
                options=targets,
                key="pv_target",
            )
            st.plotly_chart(
                ch.plot_pvalue_heatmap(
                    pvalues_df, pv_target,
                    group_col=group_col_res,
                    threshold=pvalue_threshold,
                ),
                use_container_width=True,
            )
