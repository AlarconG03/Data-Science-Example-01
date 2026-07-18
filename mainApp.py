"""
Dashboard Meteorológico por Comunas (Datos Sintéticos) - Streamlit + Plotly
------------------------------------------------------------------------------
EAFIT 2026 · Ciencia de Datos · Profesor Jorge Padilla · Julio 2026

App de una sola página con:
1. Generación de datos meteorológicos sintéticos por comuna (serie de tiempo,
   500 registros, 10 columnas de tipos mixtos) pensados para apoyar decisiones
   de la alcaldía sobre riesgos y posibles desastres.
2. Esquema de métricas cuantitativas y cualitativas.
3. Gráficas dinámicas con Plotly (variables, umbrales y personalización).
4. Panel de riesgo y alertas con umbrales interactivos.
5. Acceso protegido con código clave.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Dashboard Meteorológico · Comunas",
    page_icon="🌦️",
    layout="wide",
)

CODIGO_ACCESO = "333"

# =========================================================
# PANEL LATERAL PERSONALIZADO (IDENTIDAD DEL CURSO)
# =========================================================
def mostrar_panel_identidad():
    st.sidebar.markdown(
        """
        <div style="background-color:#003057;padding:16px;border-radius:10px;margin-bottom:16px;">
            <h2 style="color:#FFFFFF;margin:0;font-size:22px;">🎓 EAFIT 2026</h2>
            <h4 style="color:#FFD100;margin:4px 0 10px 0;">Ciencia de Datos</h4>
            <p style="color:#FFFFFF;margin:0;font-size:14px;">👨‍🏫 Profesor: Jorge Padilla</p>
            <p style="color:#FFFFFF;margin:0;font-size:14px;">🗓️ Julio 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


mostrar_panel_identidad()

# =========================================================
# CONTROL DE ACCESO POR CÓDIGO
# =========================================================
def verificar_acceso() -> bool:
    if st.session_state.get("autenticado", False):
        return True

    st.sidebar.subheader("🔒 Acceso al dashboard")

    def validar():
        if st.session_state.get("codigo_input", "") == CODIGO_ACCESO:
            st.session_state["autenticado"] = True
        else:
            st.session_state["autenticado"] = False

    st.sidebar.text_input(
        "Código de acceso", type="password", key="codigo_input", on_change=validar
    )

    if "autenticado" in st.session_state and not st.session_state["autenticado"]:
        st.sidebar.error("Código incorrecto. Intenta de nuevo.")

    if not st.session_state.get("autenticado", False):
        st.title("🌦️ Dashboard Meteorológico por Comunas")
        st.info("🔑 Ingresa el código de acceso en la barra lateral para continuar.")
        return False
    return True


if not verificar_acceso():
    st.stop()

# =========================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS METEOROLÓGICOS POR COMUNA
# =========================================================
# Comunas urbanas de referencia (nombre + población base aproximada y altitud
# base en msnm). Los valores de población y altitud son de referencia y se
# perturban aleatoriamente: el dataset es 100% sintético con fines académicos.
COMUNAS_INFO = {
    "Comuna 1 - Popular":         {"poblacion_base": 129000, "altitud_base": 1550},
    "Comuna 2 - Santa Cruz":      {"poblacion_base": 104000, "altitud_base": 1540},
    "Comuna 3 - Manrique":        {"poblacion_base": 155000, "altitud_base": 1560},
    "Comuna 4 - Aranjuez":        {"poblacion_base": 157000, "altitud_base": 1500},
    "Comuna 5 - Castilla":        {"poblacion_base": 147000, "altitud_base": 1490},
    "Comuna 6 - Doce de Octubre": {"poblacion_base": 197000, "altitud_base": 1600},
    "Comuna 7 - Robledo":         {"poblacion_base": 190000, "altitud_base": 1620},
    "Comuna 8 - Villa Hermosa":   {"poblacion_base": 154000, "altitud_base": 1650},
    "Comuna 9 - Buenos Aires":    {"poblacion_base": 144000, "altitud_base": 1580},
    "Comuna 10 - La Candelaria":  {"poblacion_base": 93000,  "altitud_base": 1495},
}
NOMBRES_COMUNAS = list(COMUNAS_INFO.keys())
DIRECCIONES_VIENTO = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

COL_NUMERICAS = [
    "temperatura_c", "humedad_relativa", "velocidad_viento_kmh",
    "precipitacion_mm", "poblacion", "altitud_msnm",
]
COL_CATEGORICAS = ["comuna", "direccion_viento", "nivel_riesgo"]
COL_FECHA = "fecha"


def calcular_nivel_riesgo(precipitacion, viento, temperatura):
    """Calcula un puntaje de riesgo (0-1) combinando lluvia, viento y temperatura
    extrema, y lo clasifica en categorías útiles para la alcaldía."""
    score_lluvia = np.clip(precipitacion / 80, 0, 1)
    score_viento = np.clip(viento / 45, 0, 1)
    score_temp = np.clip(np.abs(temperatura - 22) / 15, 0, 1)
    score = 0.5 * score_lluvia + 0.3 * score_viento + 0.2 * score_temp

    nivel = np.select(
        [score < 0.25, score < 0.5, score < 0.75],
        ["Bajo", "Medio", "Alto"],
        default="Crítico",
    )
    return nivel


def generar_datos(dias_serie: int, semilla: int) -> pd.DataFrame:
    """Simula un dataset sintético meteorológico por comuna. Genera una
    serie de tiempo diaria (dias_serie días) para cada una de las 10 comunas,
    con 10 columnas de tipos de datos mixtos."""
    rng = np.random.default_rng(semilla)

    fecha_inicio = datetime.today() - timedelta(days=dias_serie)
    fechas = pd.to_datetime([fecha_inicio + timedelta(days=d) for d in range(dias_serie)])

    filas = []
    for comuna in NOMBRES_COMUNAS:
        info = COMUNAS_INFO[comuna]

        # Estacionalidad suave con ruido: temperatura oscila alrededor de 22°C
        t = np.arange(dias_serie)
        ciclo = 2.5 * np.sin(2 * np.pi * t / 15)
        temperatura = 22 + ciclo + rng.normal(0, 1.5, dias_serie)

        humedad = np.clip(rng.normal(75, 10, dias_serie) - 0.5 * ciclo, 40, 100)

        viento = np.clip(rng.gamma(shape=2.0, scale=6.0, size=dias_serie), 0, 60)

        # Eventos de lluvia intermitentes (temporada de lluvias simulada)
        prob_lluvia = 0.35 + 0.2 * np.sin(2 * np.pi * t / 30)
        llueve = rng.random(dias_serie) < np.clip(prob_lluvia, 0.05, 0.9)
        precipitacion = np.where(llueve, rng.gamma(shape=2.0, scale=12.0, size=dias_serie), 0.0)
        precipitacion = np.clip(precipitacion, 0, 120).round(1)

        poblacion = (info["poblacion_base"] + rng.normal(0, 300, dias_serie).cumsum() / 30).astype(int)
        altitud = np.clip(info["altitud_base"] + rng.normal(0, 5, dias_serie), 1400, 1700).round(0)

        direccion_viento = rng.choice(DIRECCIONES_VIENTO, size=dias_serie)

        nivel_riesgo = calcular_nivel_riesgo(precipitacion, viento, temperatura)

        df_comuna = pd.DataFrame({
            "fecha": fechas,
            "comuna": comuna,
            "temperatura_c": temperatura.round(1),
            "humedad_relativa": humedad.round(1),
            "velocidad_viento_kmh": viento.round(1),
            "direccion_viento": direccion_viento,
            "precipitacion_mm": precipitacion,
            "poblacion": poblacion,
            "altitud_msnm": altitud,
            "nivel_riesgo": nivel_riesgo,
        })
        filas.append(df_comuna)

    df = pd.concat(filas, ignore_index=True)
    return df.sort_values(["fecha", "comuna"]).reset_index(drop=True)


def entropia(serie: pd.Series) -> float:
    p = serie.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


# =========================================================
# BARRA LATERAL: CONTROL DE SIMULACIÓN
# =========================================================
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Simulación de datos")
dias_serie = st.sidebar.slider("Días de la serie de tiempo (por comuna)", 20, 100, 50, step=5)
semilla = st.sidebar.number_input("Semilla aleatoria (seed)", value=42, min_value=0, step=1)
regenerar = st.sidebar.button("🔄 Generar / Regenerar datos")

st.sidebar.caption(f"Total de registros: **{dias_serie * len(NOMBRES_COMUNAS)}** (10 comunas × {dias_serie} días)")

necesita_generar = (
    "df" not in st.session_state
    or st.session_state.get("dias_serie") != dias_serie
    or st.session_state.get("semilla") != semilla
    or regenerar
)
if necesita_generar:
    st.session_state["df"] = generar_datos(dias_serie, semilla)
    st.session_state["dias_serie"] = dias_serie
    st.session_state["semilla"] = semilla

df = st.session_state["df"]

st.sidebar.success(f"Dataset activo: {len(df):,} registros × {df.shape[1]} columnas")
st.sidebar.download_button(
    "⬇️ Descargar dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="clima_comunas_sintetico.csv",
    mime="text/csv",
)

if st.sidebar.button("🚪 Cerrar sesión"):
    st.session_state["autenticado"] = False
    st.rerun()

# =========================================================
# ENCABEZADO
# =========================================================
st.title("🌦️ Dashboard Meteorológico por Comunas · Datos Sintéticos")
st.caption(
    "Datos simulados dentro de la app para apoyar decisiones de la alcaldía sobre riesgos "
    "climáticos (lluvias intensas, vientos fuertes, olas de calor). Ajusta la simulación desde la "
    "barra lateral y presiona **Generar / Regenerar datos**."
)

tab_datos, tab_metricas, tab_graficas, tab_riesgo = st.tabs(
    ["📄 Datos", "📊 Métricas", "📈 Gráficas dinámicas", "🚨 Riesgo y Alertas"]
)

# =========================================================
# TAB 1: DATOS
# =========================================================
with tab_datos:
    st.subheader("Vista previa del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Rango de fechas", f"{df[COL_FECHA].min().date()} → {df[COL_FECHA].max().date()}")
    c4.metric("Comunas", df["comuna"].nunique())

    st.dataframe(df.head(200), use_container_width=True)

    with st.expander("Tipos de datos por columna"):
        tipos = pd.DataFrame({
            "columna": df.columns,
            "tipo": df.dtypes.astype(str).values,
            "valores_unicos": [df[c].nunique() for c in df.columns],
            "nulos": df.isna().sum().values,
        })
        st.dataframe(tipos, use_container_width=True, hide_index=True)

# =========================================================
# TAB 2: MÉTRICAS (cuantitativas + cualitativas)
# =========================================================
with tab_metricas:
    st.subheader("Estadísticas cuantitativas")
    col_num = st.selectbox("Selecciona una variable numérica", COL_NUMERICAS)
    serie_num = df[col_num]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Media", f"{serie_num.mean():.2f}")
    m2.metric("Mediana", f"{serie_num.median():.2f}")
    m3.metric("Desv. estándar", f"{serie_num.std():.2f}")
    m4.metric("Varianza", f"{serie_num.var():.2f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Mínimo", f"{serie_num.min():.2f}")
    m6.metric("Máximo", f"{serie_num.max():.2f}")
    m7.metric("Asimetría (skew)", f"{serie_num.skew():.2f}")
    m8.metric("Curtosis", f"{serie_num.kurtosis():.2f}")

    percentil = st.slider("Percentil a consultar", 1, 99, 90)
    st.info(f"Percentil {percentil} de **{col_num}**: {np.percentile(serie_num, percentil):.2f}")

    with st.expander("Resumen estadístico completo (describe)"):
        st.dataframe(df[COL_NUMERICAS].describe().T, use_container_width=True)

    st.divider()
    st.subheader("Estadísticas cualitativas")
    col_cat = st.selectbox("Selecciona una variable categórica", COL_CATEGORICAS)
    conteo = df[col_cat].value_counts()
    proporcion = df[col_cat].value_counts(normalize=True) * 100

    q1, q2, q3 = st.columns(3)
    q1.metric("Moda", df[col_cat].mode()[0])
    q2.metric("Categorías únicas", df[col_cat].nunique())
    q3.metric("Entropía (bits)", f"{entropia(df[col_cat]):.2f}")

    tabla_cat = pd.DataFrame({
        "conteo": conteo,
        "porcentaje (%)": proporcion.round(2),
    })
    st.dataframe(tabla_cat, use_container_width=True)

    st.divider()
    st.subheader("Tabla cruzada (categórica vs categórica)")
    c1, c2 = st.columns(2)
    cat_a = c1.selectbox("Variable A", COL_CATEGORICAS, index=0, key="cruce_a")
    cat_b = c2.selectbox("Variable B", COL_CATEGORICAS, index=2, key="cruce_b")
    if cat_a != cat_b:
        cruce = pd.crosstab(df[cat_a], df[cat_b])
        fig_cruce = px.imshow(
            cruce, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(color="Conteo"),
        )
        st.plotly_chart(fig_cruce, use_container_width=True)
    else:
        st.warning("Selecciona dos variables diferentes para cruzarlas.")

# =========================================================
# TAB 3: GRÁFICAS DINÁMICAS (PLOTLY)
# =========================================================
with tab_graficas:
    st.subheader("Explorador gráfico interactivo")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
    tipo_grafico = ctrl1.selectbox(
        "Tipo de gráfica",
        ["Histograma", "Boxplot", "Violín", "Barras (conteo/promedio)",
         "Dispersión (Scatter)", "Serie temporal", "Mapa de calor - Correlación"],
    )
    paleta_nombre = ctrl2.selectbox(
        "Paleta de colores", ["Plotly", "D3", "G10", "T10", "Pastel", "Set2", "Dark24"]
    )
    paleta_map = {
        "Plotly": px.colors.qualitative.Plotly, "D3": px.colors.qualitative.D3,
        "G10": px.colors.qualitative.G10, "T10": px.colors.qualitative.T10,
        "Pastel": px.colors.qualitative.Pastel, "Set2": px.colors.qualitative.Set2,
        "Dark24": px.colors.qualitative.Dark24,
    }
    paleta = paleta_map[paleta_nombre]
    titulo_custom = ctrl3.text_input("Título personalizado", value="")

    fig = None
    usa_eje_y_numerico = True  # controla si tiene sentido ofrecer línea de umbral

    if tipo_grafico == "Histograma":
        col1, col2 = st.columns(2)
        var = col1.selectbox("Variable numérica", COL_NUMERICAS)
        bins = col2.slider("Número de bins", 5, 100, 30)
        color_por = st.selectbox("Colorear por (opcional)", ["Ninguno"] + COL_CATEGORICAS)
        fig = px.histogram(
            df, x=var, nbins=bins,
            color=None if color_por == "Ninguno" else color_por,
            color_discrete_sequence=paleta,
        )

    elif tipo_grafico == "Boxplot":
        col1, col2 = st.columns(2)
        var_y = col1.selectbox("Variable numérica (Y)", COL_NUMERICAS)
        var_x = col2.selectbox("Agrupar por (X, opcional)", ["Ninguno"] + COL_CATEGORICAS)
        fig = px.box(
            df, x=None if var_x == "Ninguno" else var_x, y=var_y,
            color=None if var_x == "Ninguno" else var_x,
            color_discrete_sequence=paleta,
        )

    elif tipo_grafico == "Violín":
        col1, col2 = st.columns(2)
        var_y = col1.selectbox("Variable numérica (Y)", COL_NUMERICAS)
        var_x = col2.selectbox("Agrupar por (X, opcional)", ["Ninguno"] + COL_CATEGORICAS)
        fig = px.violin(
            df, x=None if var_x == "Ninguno" else var_x, y=var_y, box=True, points=False,
            color=None if var_x == "Ninguno" else var_x,
            color_discrete_sequence=paleta,
        )

    elif tipo_grafico == "Barras (conteo/promedio)":
        col1, col2, col3 = st.columns(3)
        var_cat = col1.selectbox("Variable categórica (X)", COL_CATEGORICAS)
        agregacion = col2.selectbox("Agregación", ["Conteo", "Promedio de variable numérica"])
        if agregacion == "Conteo":
            datos_barras = df[var_cat].value_counts().reset_index()
            datos_barras.columns = [var_cat, "conteo"]
            fig = px.bar(datos_barras, x=var_cat, y="conteo", color=var_cat,
                         color_discrete_sequence=paleta)
        else:
            var_num = col3.selectbox("Variable numérica", COL_NUMERICAS)
            datos_barras = df.groupby(var_cat)[var_num].mean().reset_index()
            fig = px.bar(datos_barras, x=var_cat, y=var_num, color=var_cat,
                         color_discrete_sequence=paleta)

    elif tipo_grafico == "Dispersión (Scatter)":
        col1, col2, col3 = st.columns(3)
        var_x = col1.selectbox("Eje X", COL_NUMERICAS, index=0)
        var_y = col2.selectbox("Eje Y", COL_NUMERICAS, index=3)
        color_por = col3.selectbox("Colorear por (opcional)", ["Ninguno"] + COL_CATEGORICAS)
        fig = px.scatter(
            df, x=var_x, y=var_y,
            color=None if color_por == "Ninguno" else color_por,
            opacity=0.6, color_discrete_sequence=paleta,
        )

    elif tipo_grafico == "Serie temporal":
        col1, col2, col3 = st.columns(3)
        var_num = col1.selectbox("Variable numérica", COL_NUMERICAS)
        comunas_sel = col2.multiselect("Comunas a mostrar", NOMBRES_COMUNAS, default=NOMBRES_COMUNAS[:3])
        agregacion = col3.selectbox("Agregación diaria", ["Sin agregar (por comuna)", "Promedio ciudad"])
        df_ts = df if not comunas_sel else df[df["comuna"].isin(comunas_sel)]
        if agregacion == "Promedio ciudad":
            serie = df_ts.groupby(COL_FECHA)[var_num].mean().reset_index()
            fig = px.line(serie, x=COL_FECHA, y=var_num, markers=True, color_discrete_sequence=paleta)
        else:
            fig = px.line(df_ts, x=COL_FECHA, y=var_num, color="comuna", markers=True,
                         color_discrete_sequence=paleta)

    elif tipo_grafico == "Mapa de calor - Correlación":
        corr = df[COL_NUMERICAS].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        usa_eje_y_numerico = False

    # ---- Umbral (línea de referencia) ----
    if usa_eje_y_numerico and fig is not None:
        st.markdown("**Línea de umbral (opcional)**")
        u1, u2 = st.columns([1, 3])
        mostrar_umbral = u1.checkbox("Mostrar umbral")
        if mostrar_umbral:
            valor_umbral = u2.number_input("Valor del umbral", value=float(df[COL_NUMERICAS[0]].mean()))
            fig.add_hline(
                y=valor_umbral, line_dash="dash", line_color="red",
                annotation_text=f"Umbral: {valor_umbral}", annotation_position="top left",
            )

    if fig is not None:
        if titulo_custom:
            fig.update_layout(title=titulo_custom)
        fig.update_layout(template="plotly_white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4: RIESGO Y ALERTAS (apoyo a decisiones de la alcaldía)
# =========================================================
with tab_riesgo:
    st.subheader("Panel de riesgo y alertas tempranas")
    st.caption(
        "Define umbrales de precipitación y viento para identificar comunas en alerta. "
        "Estos criterios pueden ajustarse según los protocolos de gestión del riesgo de la alcaldía."
    )

    u1, u2 = st.columns(2)
    umbral_precipitacion = u1.slider("Umbral de precipitación (mm)", 0, 120, 60, step=5)
    umbral_viento = u2.slider("Umbral de velocidad del viento (km/h)", 0, 60, 35, step=5)

    df_alerta = df[
        (df["precipitacion_mm"] >= umbral_precipitacion) | (df["velocidad_viento_kmh"] >= umbral_viento)
    ]

    a1, a2, a3 = st.columns(3)
    a1.metric("Registros en alerta", len(df_alerta))
    a2.metric("% del total", f"{100 * len(df_alerta) / len(df):.1f}%")
    a3.metric("Comunas afectadas", df_alerta["comuna"].nunique())

    if len(df_alerta) > 0:
        conteo_comuna = df_alerta["comuna"].value_counts().reset_index()
        conteo_comuna.columns = ["comuna", "eventos_alerta"]
        fig_alerta = px.bar(
            conteo_comuna, x="comuna", y="eventos_alerta", color="comuna",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Eventos en alerta por comuna",
        )
        fig_alerta.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_alerta, use_container_width=True)

        st.markdown("**Distribución del nivel de riesgo (dataset completo)**")
        conteo_riesgo = df["nivel_riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto", "Crítico"]).fillna(0)
        fig_riesgo = px.bar(
            x=conteo_riesgo.index, y=conteo_riesgo.values,
            color=conteo_riesgo.index,
            color_discrete_map={"Bajo": "#2ECC71", "Medio": "#F1C40F", "Alto": "#E67E22", "Crítico": "#E74C3C"},
            labels={"x": "Nivel de riesgo", "y": "Cantidad de registros"},
        )
        fig_riesgo.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_riesgo, use_container_width=True)

        st.markdown("**Registros en alerta**")
        st.dataframe(
            df_alerta.sort_values("fecha", ascending=False)[
                ["fecha", "comuna", "precipitacion_mm", "velocidad_viento_kmh", "temperatura_c", "nivel_riesgo"]
            ],
            use_container_width=True,
        )
    else:
        st.success("No hay registros que superen los umbrales definidos. ✅")
