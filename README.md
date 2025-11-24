# 🌸 Clasificación de Flores Iris con Machine Learning y Streamlit

Este proyecto implementa un modelo de **clasificación supervisada** utilizando el famoso dataset **Iris**, con el objetivo de predecir la especie de una flor basándose en sus características morfológicas.  
Además, incluye un **panel interactivo en Streamlit** que permite visualizar el análisis, las métricas del modelo y realizar predicciones en tiempo real.

---

## 🚀 Objetivos del Proyecto

1. Entrenar un modelo capaz de clasificar flores Iris según sus características:
   - Longitud del sépalo
   - Ancho del sépalo
   - Longitud del pétalo
   - Ancho del pétalo

2. Crear un panel interactivo que permite:
   - Ver métricas del modelo (Exactitud, Precision, Recall y F1-score)
   - Realizar predicciones ingresando valores manualmente
   - Ver la predicción dentro de un diagrama 3D junto a los datos reales
   - Analizar visualizaciones adicionales del dataset

---

## 📊 Dataset

Se utiliza el dataset **Iris.csv**, que contiene 150 muestras de tres especies:
- *Iris-setosa*
- *Iris-versicolor*
- *Iris-virginica*

El archivo incluye:
- 4 características numéricas
- 1 variable objetivo

---

## 🔧 Tecnologías Utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib / Seaborn
- Streamlit

---

## 🧠 Modelo de Machine Learning

El modelo utilizado p es **Random Forest**

Pipeline general:
1. Carga y exploración de datos
2. Preprocesamiento
3. Entrenamiento
4. Evaluación (accuracy, precision, recall, f1)
5. Predicción con entrada del usuario

---

## 🖥️ Panel Interactivo (Streamlit)

El archivo **app.py** permite:
- Visualizar métricas del modelo
- Realizar predicciones ingresando 4 parámetros
- Ver la predicción en un gráfico **3D interactivo**
- Explorar histogramas, matrices de dispersión y otros gráficos

---


