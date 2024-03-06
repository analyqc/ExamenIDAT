import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Cargar el modelo entrenado
with open('modelo_optimizado.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
type_gaming = st.selectbox('¿Es Gaming?', [0, 1])
weight = st.number_input('Peso', min_value=0.5, max_value=5.0, value=1.5)
type_notebook = st.selectbox('¿Es Notebook?', [0, 1])
so_brand = st.selectbox('Marca del Sistema Operativo', [1, 2, 3, 4, 5])
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
has_ssd = st.selectbox('¿Tiene SSD?', [0, 1])
processor_brand = st.selectbox('Marca del Procesador', [1, 2, 3, 4, 5])
memory_gb = st.number_input('Memoria (GB)', min_value=1, max_value=1024, value=128)
screen_resolution = st.number_input('Resolución de Pantalla', min_value=800, max_value=10000000, value=4096000)

# Control de entrada para el tamaño de la pantalla en pulgadas
try:
    inches = float(st.text_input('Tamaño de Pantalla (pulgadas)', '13.3'))
except ValueError:
    st.error('Por favor, ingrese un valor numérico para el tamaño de pantalla.')

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[ram, type_gaming, weight, type_notebook, so_brand, ghz, has_ssd, processor_brand, memory_gb, screen_resolution, inches]],
                    columns=['Ram', 'TypeName_Gaming', 'Weight', 'TypeName_Notebook', 'SO_brand', 'GHz', 'has_SSD', 'Processor_brand', 'Memory_GB', 'ScreenResolution', 'Inches'])

    # Codificación One-Hot para las variables categóricas
    encoder = OneHotEncoder(drop='first', sparse=False)
    input_encoded = encoder.fit_transform(input_data[['TypeName_Gaming', 'TypeName_Notebook', 'SO_brand', 'has_SSD', 'Processor_brand']])

    # Crear DataFrame con las variables codificadas
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names(['TypeName_Gaming', 'TypeName_Notebook', 'SO_brand', 'has_SSD', 'Processor_brand']))

    # Unir el DataFrame codificado con el resto de las características numéricas
    input_data = pd.concat([input_data[['Ram', 'Weight', 'GHz', 'Memory_GB', 'ScreenResolution', 'Inches']], input_encoded_df], axis=1)

    # Asegurarse de que todas las características estén presentes
    expected_features = ['Ram', 'Weight', 'GHz', 'Memory_GB', 'ScreenResolution', 'Inches'] + list(input_encoded_df.columns)
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Estandarización de las características
    scaler = StandardScaler()

    # Realizar predicción
    try:
        input_scaled = scaler.fit_transform(input_data)
        prediction = modelo.predict(input_scaled)
        st.write(f'Precio predecido: {prediction[0]:.2f} euros')
    except ValueError as e:
        st.error(f'Error al estandarizar los datos: {e}')
