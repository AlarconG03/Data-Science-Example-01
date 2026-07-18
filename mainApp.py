"""
Dashboard COVID-19 (Datos Sintéticos) - Streamlit + Plotly
------------------------------------------------------------
App de una sola página con:
1. Generación de datos sintéticos (10.000 registros, 8 columnas, tipos mixtos)
2. Esquema de métricas cuantitativas y cualitativas
3. Gráficas dinámicas con Plotly (variables, umbrales y personalización)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Dashboard COVID-19 · Datos Sintéticos",
    page_icon="🦠",
    layout="wide",
)

DEPARTAMENTOS = [
    "Antioquia", "Bogotá D.C.", "Valle del Cauca", "Atlántico", "Santander",
    "Cundinamarca", "Bolívar", "Nariño", "Córdoba", "Tolima",
]
ESTADOS = ["Leve", "Moderado", "Grave", "Fallecido", "Recuperado"]
ESTADOS_PROB = [0.55, 0.25, 0.12, 0.03, 0.05]

COL_NUMERICAS = ["edad", "sintomas_dias", "num_comorbilidades", "saturacion_oxigeno"]
COL_CATEGORICAS = ["sexo", "departamento", "estado_clinico"]
COL_FECHA = "fecha_reporte"


# =========================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# =========================================================
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Simula un dataset sintético de casos COVID-19 con 8 columnas de tipos mixtos."""
    rng = np.random.default_rng(semilla)

    fecha_inicio = datetime(2022, 1, 1)
    dias_rango = 730
    fecha_reporte = pd.to_datetime(
        [fecha_inicio + timedelta(days=int(d)) for d in rng.integers(0, dias_rango, n_registros)]
    )

    edad = rng.normal(40, 20, n_registros).clip(0, 100).astype(int)

    sexo = rng.choice(["Masculino", "Femenino"], size=n_registros, p=[0.49, 0.51])

    departamento = rng.choice(DEPARTAMENTOS, size=n_registros)

    sintomas_dias = rng.poisson(7, n_registros).clip(0, 30).astype(int)

    estado_clinico = rng.choice(ESTADOS, size=n_registros, p=ESTADOS_PROB)

    num_comorbilidades = rng.choice(
        [0, 1, 2, 3, 4, 5], size=n_registros, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]
    ).astype(int)

    saturacion_base = rng.normal(96, 3, n_registros)
    penalizacion = np.where(
        np.isin(estado_clinico, ["Grave", "Fallecido"]), rng.uniform(8, 20, n_registros),
        np.where(estado_clinico == "Moderado", rng.uniform(2, 6, n_registros), 0.0),
    )
    saturacion_oxigeno = (saturacion_base - penalizacion).clip(70, 100).round(1)

    df = pd.DataFrame({
        "fecha_reporte": fecha_reporte,
        "edad": edad,
        "sexo": sexo,
        "departamento": departamento,
        "sintomas_dias": sintomas_dias,
        "estado_clinico": estado_clinico,
        "num_comorbilidades": num_comorbilidades,
        "saturacion_oxigeno": saturacion_oxigeno,
    })
    return df.sort_values("fecha_reporte").reset_index(drop=True)


def entropia(serie: pd.Series) -> float:
    p = serie.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


# =========================================================
# BARRA LATERAL: CONTROL DE SIMULACIÓN
# =========================================================
st.sidebar.title("⚙️ Simulación de datos")
n_registros = st.sidebar.slider("Número de registros", 1000, 20000, 10000, step=500)
semilla = st.sidebar.number_input("Semilla aleatoria (seed)", value=42, min_value=0, step=1)
regenerar = st.sidebar.button("🔄 Generar / Regenerar datos")

necesita_generar = (
    "df" not in st.session_state
    or st.session_state.get("n_registros") != n_registros
    or st.session_state.get("semilla") != semilla
    or regenerar
)
if necesita_generar:
    st.session_state["df"] = generar_datos(n_registros, semilla)
    st.session_state["n_registros"] = n_registros
    st.session_state["semilla"] = semilla

df = st.session_state["df"]

st.sidebar.success(f"Dataset activo: {len(df):,} registros × {df.shape[1]} columnas")
st.sidebar.download_button(
    "⬇️ Descargar dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="covid_sintetico.csv",
    mime="text/csv",
)

# =========================================================
# ENCABEZADO
# =========================================================
st.title("🦠 Dashboard COVID-19 · Datos Sintéticos")
st.caption(
    "Los datos se simulan dentro de la propia app (no provienen de archivos externos). "
    "Ajusta el número de registros o la semilla en la barra lateral y presiona **Generar / Regenerar datos**."
)

tab_datos, tab_metricas, tab_graficas = st.tabs(["📄 Datos", "📊 Métricas", "📈 Gráficas dinámicas"])

# =========================================================
# TAB 1: DATOS
# =========================================================
with tab_datos:
    st.subheader("Vista previa del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Rango de fechas", f"{df[COL_FECHA].min().date()} → {df[COL_FECHA].max().date()}")
    c4.metric("Departamentos", df["departamento"].nunique())

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
            opacity=0.5, color_discrete_sequence=paleta,
        )

    elif tipo_grafico == "Serie temporal":
        col1, col2, col3 = st.columns(3)
        var_num = col1.selectbox("Variable numérica", COL_NUMERICAS)
        frecuencia = col2.selectbox("Frecuencia de agregación", ["Diaria", "Semanal", "Mensual"])
        agregacion = col3.selectbox("Agregación", ["Promedio", "Suma", "Conteo"])
        freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}
        serie = df.set_index(COL_FECHA)[var_num].resample(freq_map[frecuencia])
        serie = {"Promedio": serie.mean(), "Suma": serie.sum(), "Conteo": serie.count()}[agregacion]
        serie = serie.reset_index()
        fig = px.line(serie, x=COL_FECHA, y=var_num, markers=True,
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
