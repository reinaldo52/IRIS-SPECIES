import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ===============================
# Cargar datos
# ===============================
df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Preparar datos
X = df.drop("Species", axis=1)
y = df["Species"]

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# Métricas
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# ===============================
# INTERFAZ STREAMLIT
# ===============================
st.title("🌸 Panel Interactivo - Clasificación Iris con Random Forest")
st.write("Este panel permite explorar el dataset Iris, visualizar el modelo y realizar predicciones interactivas.")

st.header("📊 Métricas del Modelo")
st.write(f"**Exactitud:** {accuracy:.4f}")
st.write(f"**Precisión:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")

st.header("📈 Visualizaciones del Dataset")

# Histograma
col1, col2 = st.columns(2)
with col1:
    st.subheader("Histograma por característica")
    feature = st.selectbox("Seleccione una variable:", X.columns)
    fig1 = px.histogram(df, x=feature, color="Species", nbins=20)
    st.plotly_chart(fig1)

# Scatter matrix
with col2:
    st.subheader("Matriz de dispersión")
    fig2 = px.scatter_matrix(df, dimensions=X.columns, color="Species")
    st.plotly_chart(fig2)

# ===============================
# Predicción interactiva
# ===============================
st.header("🔮 Predicción interactiva de especie")

sl = st.number_input("Longitud del sépalo (cm)", min_value=0.0, max_value=10.0, step=0.1)
sw = st.number_input("Anchura del sépalo (cm)", min_value=0.0, max_value=10.0, step=0.1)
pl = st.number_input("Longitud del pétalo (cm)", min_value=0.0, max_value=10.0, step=0.1)
pw = st.number_input("Anchura del pétalo (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Crear muestra
new_point = np.array([[sl, sw, pl, pw]])

if st.button("Predecir especie"):
    pred_encoded = model.predict(new_point)[0]
    pred_species = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"🌼 **La especie predicha es:** {pred_species}")

# ===============================
# Visualización 3D con punto predicho
# ===============================
st.header("🧭 Posición de la nueva muestra en un espacio 3D")

x_axis = st.selectbox("Eje X", X.columns, index=0)
y_axis = st.selectbox("Eje Y", X.columns, index=1)
z_axis = st.selectbox("Eje Z", X.columns, index=2)

fig3 = px.scatter_3d(
    df,
    x=x_axis,
    y=y_axis,
    z=z_axis,
    color="Species",
    title="Representación 3D del dataset"
)

# Agregar el punto predicho
if sl != 0 or sw != 0 or pl != 0 or pw != 0:
    fig3.add_scatter3d(
        x=[sl],
        y=[sw],
        z=[pl],
        mode='markers',
        marker=dict(size=8),
        name="Nueva muestra"
    )

st.plotly_chart(fig3)

st.write("Panel creado por Streamlit – Proyecto de clasificación Iris 🚀")
