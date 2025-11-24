import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Iris Classifier - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌸 Panel Interactivo: Clasificación de Especies (Iris)")

# -----------------------------
# Cargar dataset (ruta local subida)
# -----------------------------
DATA_PATH = "/mnt/data/Iris.csv"  # <-- usa esta ruta si trabajas en Colab
# Si lo despliegas en Streamlit Cloud usa: df = pd.read_csv("Iris.csv")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(
        f"No se pudo cargar el dataset en {DATA_PATH}. Asegúrate de que el archivo exista. Error: {e}"
    )
    st.stop()

# Asegurarse de las columnas esperadas (robustez)
expected_cols = {"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"El archivo no contiene las columnas esperadas: {expected_cols}. Columnas encontradas: {list(df.columns)}")
    st.stop()

# -----------------------------
# Preparar datos y entrenar modelo
# -----------------------------
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[FEATURES]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción de test para métricas
y_pred = model.predict(X_test)

# -----------------------------
# Métricas del modelo
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

# -----------------------------
# Layout: sidebar para predicción
# -----------------------------
st.sidebar.header("🔎 Predicción de Especie (entrada)")
st.sidebar.write("Ingresa las medidas (cm) dentro de los rangos permitidos:")

# Rangos basados en dataset
min_max = {
    "SepalLengthCm": (float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max())),
    "SepalWidthCm": (float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max())),
    "PetalLengthCm": (float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max())),
    "PetalWidthCm": (float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max())),
}

sepal_length = st.sidebar.number_input(
    "Longitud del sépalo (SepalLengthCm)",
    min_value=round(min_max["SepalLengthCm"][0], 1),
    max_value=round(min_max["SepalLengthCm"][1], 1),
    value=round(df["SepalLengthCm"].median(), 1),
    step=0.1,
)

sepal_width = st.sidebar.number_input(
    "Anchura del sépalo (SepalWidthCm)",
    min_value=round(min_max["SepalWidthCm"][0], 1),
    max_value=round(min_max["SepalWidthCm"][1], 1),
    value=round(df["SepalWidthCm"].median(), 1),
    step=0.1,
)

petal_length = st.sidebar.number_input(
    "Longitud del pétalo (PetalLengthCm)",
    min_value=round(min_max["PetalLengthCm"][0], 1),
    max_value=round(min_max["PetalLengthCm"][1], 1),
    value=round(df["PetalLengthCm"].median(), 1),
    step=0.1,
)

petal_width = st.sidebar.number_input(
    "Anchura del pétalo (PetalWidthCm)",
    min_value=round(min_max["PetalWidthCm"][0], 1),
    max_value=round(min_max["PetalWidthCm"][1], 1),
    value=round(df["PetalWidthCm"].median(), 1),
    step=0.1,
)

predict_button = st.sidebar.button("🔍 Predecir especie")

# -----------------------------
# Main: métricas y visualizaciones
# -----------------------------
with st.expander("📊 Métricas del modelo (detalles)", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision (weighted)", f"{precision:.3f}")
    col3.metric("Recall (weighted)", f"{recall:.3f}")
    col4.metric("F1-score (weighted)", f"{f1:.3f}")

    st.markdown("**Matriz de confusión (conjunto de prueba)**")
    labels = list(np.unique(y))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(x="Predicción", y="Verdadero", color="Conteo"),
        x=labels,
        y=labels,
        title="Matriz de Confusión"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# Visualizaciones principales
st.header("📈 Visualizaciones del Dataset")

# Histogramas (seleccionar característica)
col_h1, col_h2 = st.columns([1, 2])
with col_h1:
    feature = st.selectbox("Seleccione característica para histograma", FEATURES, index=0)
    nbins = st.slider("Número de bins", 5, 50, 20)

    fig_hist = px.histogram(df, x=feature, color="Species", nbins=nbins,
                            title=f"Histograma de {feature} por especie")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_h2:
    st.markdown("**Matriz de dispersión (Scatter Matrix)**")
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=FEATURES,
        color="Species",
        title="Matriz de dispersión de características"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Distribución conjunta (opcional)
st.markdown("---")
st.subheader("🔎 Relación entre dos características")
col_a, col_b = st.columns(2)
with col_a:
    x_feat = st.selectbox("Eje X", FEATURES, index=0, key="x_feat")
with col_b:
    y_feat = st.selectbox("Eje Y", FEATURES, index=2, key="y_feat")

fig2 = px.scatter(
    df, x=x_feat, y=y_feat, color="Species", title=f"{x_feat} vs {y_feat}",
    hover_data=FEATURES
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Predicción y gráfico 3D con nueva muestra
# -----------------------------
if predict_button:
    # Conversión segura a float
    try:
        user_point = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    except Exception as e:
        st.error("Los valores ingresados no son válidos. Asegúrate de introducir números.")
        st.stop()

    # Predecir especie y probabilidades
    pred = model.predict(user_point)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(user_point)[0]
        # construir dict species -> prob
        proba_dict = dict(zip(model.classes_, proba))
    else:
        proba_dict = {}

    st.success(f"🌼 Especie predicha: **{pred}**")
    if proba is not None:
        st.info("Probabilidades por clase:")
        st.table(pd.DataFrame([proba_dict]).T.rename(columns={0: "Probabilidad"}))

    # Gráfico 3D: SepalLength, SepalWidth, PetalLength (puedes cambiar ejes si quieres)
    st.subheader("📌 Muestra posicionada en el espacio 3D")
    df_plot = df.copy()
    df_plot["is_user"] = "Dataset"
    user_df = pd.DataFrame({
        "SepalLengthCm": [user_point[0, 0]],
        "SepalWidthCm": [user_point[0, 1]],
        "PetalLengthCm": [user_point[0, 2]],
        "PetalWidthCm": [user_point[0, 3]],
        "Species": [pred],
        "is_user": ["Nueva muestra"]
    })
    df_plot = pd.concat([df_plot, user_df], ignore_index=True)

    fig_3d = px.scatter_3d(
        df_plot,
        x="SepalLengthCm",
        y="SepalWidthCm",
        z="PetalLengthCm",
        color="Species",
        symbol="is_user",
        title="Visualización 3D: SepalLength, SepalWidth, PetalLength"
    )

    # Hacer el punto del usuario más grande y con borde
    # Encontrar índice del usuario (última fila)
    fig_3d.update_traces(marker=dict(size=5))
    # Añadir la nueva muestra como un trace separado (para controlar tamaño/color)
    fig_3d.add_trace(
        go.Scatter3d(
            x=[user_point[0, 0]],
            y=[user_point[0, 1]],
            z=[user_point[0, 2]],
            mode="markers",
            marker=dict(size=10, color="black", symbol="diamond"),
            name="Nueva muestra",
            showlegend=True,
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------
# Pie / Footer
# -----------------------------
st.markdown("---")
st.markdown("**Notas:** Los rangos de entrada están limitados a los mínimos y máximos observados en el dataset para evitar entradas inválidas. Si despliegas en Streamlit Cloud, sube `Iris.csv` al repo y cambia DATA_PATH a 'Iris.csv'.")
st.markdown("Hecho con ❤️")


