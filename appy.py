import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# -----------------------------
#   CARGA DE DATOS
# -----------------------------
df = pd.read_csv("Iris.csv")

st.title("🌸 Panel Interactivo de Clasificación de Flores – Iris Dataset")

st.write("""
Este panel permite:
- Visualizar el análisis del dataset  
- Consultar métricas del modelo  
- Ingresar valores para predecir la especie  
- Ver la predicción en un **gráfico 3D interactivo**
""")

# -----------------------------
#   ENTRENAMIENTO DEL MODELO
# -----------------------------
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Reporte de métricas
report = classification_report(y_test, y_pred, output_dict=True)

# -----------------------------
#      SECCIÓN: MÉTRICAS
# -----------------------------
st.header("📊 Métricas del Modelo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
col3.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
col4.metric("F1-score", f"{report['weighted avg']['f1-score']:.3f}")

# -----------------------------
#   SECCIÓN: HISTOGRAMAS
# -----------------------------
st.header("📈 Histogramas de las Variables")

feature = st.selectbox(
    "Seleccione una característica",
    df.columns[:-1]
)

fig_hist = px.histogram(df, x=feature, color="Species", nbins=20, title=f"Histograma de {feature}")
st.plotly_chart(fig_hist)

# -----------------------------
#    MATRIZ DE DISPERSIÓN
# -----------------------------
st.header("🔍 Matriz de Dispersión")

fig_scatter = px.scatter_matrix(
    df,
    dimensions=df.columns[:-1],
    color="Species",
    title="Matriz de dispersión del dataset Iris"
)
st.plotly_chart(fig_scatter)

# -----------------------------
# PREDICCIÓN DEL USUARIO
# -----------------------------
st.header("🌼 Predicción de Especie")

sepal_length = st.number_input("Longitud del sépalo", min_value=0.0, step=0.1)
sepal_width = st.number_input("Anchura del sépalo", min_value=0.0, step=0.1)
petal_length = st.number_input("Longitud del pétalo", min_value=0.0, step=0.1)
petal_width = st.number_input("Anchura del pétalo", min_value=0.0, step=0.1)

user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predecir especie"):
    prediction = model.predict(user_data)[0]
    st.success(f"🌸 **Especie Predicha:** {prediction}")

    # -----------------------------
    #   GRAFICA 3D CON PUNTO NUEVO
    # -----------------------------
    st.subheader("📌 Muestra posicionada en el espacio 3D")

    df_temp = df.copy()
    df_temp["is_user"] = "Dataset"
    user_row = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width],
        "Species": [prediction],
        "is_user": ["Nueva muestra"]
    })

    df_plot = pd.concat([df_temp, user_row])

    fig_3d = px.scatter_3d(
        df_plot,
        x="SepalLengthCm",
        y="SepalWidthCm",
        z="PetalLengthCm",
        color="Species",
        symbol="is_user",
        size=[10 if x=="Nueva muestra" else 4 for x in df_plot["is_user"]],
        title="Visualización 3D con la muestra ingresada"
    )

    st.plotly_chart(fig_3d)

