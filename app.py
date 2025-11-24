import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Cargar dataset
# ---------------------------
df = pd.read_csv("Iris.csv")

# Seleccionar columnas
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# Interfaz en Streamlit
# ---------------------------
st.title("🌸 Clasificación de Especies de Iris")
st.write("Ingresa los valores para predecir la especie:")

# RANGOS OFICIALES DEL DATASET
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=4.3,
    max_value=7.9,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.4,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=6.9,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=0.2,
    step=0.1
)

# Predicción
if st.button("🔍 Predecir"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    st.success(f"La especie predicha es: **{prediction}**")

