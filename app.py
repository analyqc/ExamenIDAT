import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
with open('modelo_optimizado.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
type_gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
weight = st.number_input('Peso', min_value=0.5, max_value=5.0, value=1.5)
type_notebook = st.selectbox('¿Es Notebook?', ['No', 'Sí'])
so_brand = st.selectbox('Marca del Sistema Operativo', ['1', '2', '3', '4', '5'])  # Reemplaza con las marcas reales
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
has_ssd = st.selectbox('¿Tiene SSD?', ['No', 'Sí'])
processor_brand = st.selectbox('Marca del Procesador', ['1', '2', '3', '4', '5'])  # Reemplaza con las marcas reales
memory_gb = st.number_input('Memoria (GB)', min_value=1, max_value=1024, value=128)
screen_resolution = st.number_input('Resolución de Pantalla', min_value=800, max_value=10000000, value=4096000)
inches = st.number_input('Tamaño de Pantalla (pulgadas)', min_value=10, max_value=20, value=13.3)

# Convertir entradas a formato numérico
type_gaming = 1 if type_gaming == 'Sí' else 0
type_notebook = 1 if type_notebook == 'Sí' else 0
has_ssd = 1 if has_ssd == 'Sí' else 0

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[ram, type_gaming, weight, type_notebook, so_brand, ghz, has_ssd, processor_brand, memory_gb, screen_resolution, inches]],
                    columns=['Ram', 'TypeName_Gaming', 'Weight', 'TypeName_Notebook', 'SO_brand', 'GHz', 'has_SSD', 'Processor_brand', 'Memory_GB', 'ScreenResolution', 'Inches'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    prediction = modelo.predict(input_scaled)

    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} ')
