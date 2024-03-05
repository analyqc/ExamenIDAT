import streamlit as st
import pandas as pd
import numpy as np
import pickleimport streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Función para obtener la marca del procesador
def get_processor_brand(cpu):
    if 'Intel' in cpu:
        return 1
    elif 'AMD' in cpu:
        return 2
    else:
        return 3

# Función para obtener la marca del sistema operativo
def get_processor_SO(so):
    lowercase_so = so.lower()
    
    conditions = [
        ('windows', 1),
        ('no os', 2),
        ('linux', 3),
        ('chrome os', 4),
        ('mac', 5),
        ('android', 6)
    ]

    for condition, result in conditions:
        if condition in lowercase_so:
            return result
    return 0

# Cargar el modelo entrenado
with open('modelo_optimizado.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
typename_gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
weight = st.number_input('Peso', min_value=0.5, max_value=5.0, value=1.37)
typename_notebook = st.selectbox('¿Es Notebook?', ['No', 'Sí'])
so_brand = st.text_input('Marca del SO', 'Windows')
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
has_ssd = st.selectbox('¿Tiene SSD?', ['No', 'Sí'])
processor_brand = st.text_input('Marca del Procesador', 'Intel')
memory_gb = st.number_input('Memory (GB)', min_value=1, max_value=1024, value=128)
screen_resolution = st.number_input('Screen Resolution', min_value=1, max_value=10000000, value=4096000)
inches = st.number_input('Inches', min_value=10, max_value=20, value=13.3)

# Convertir entradas a formato numérico
typename_gaming = 1 if typename_gaming == 'Sí' else 0
typename_notebook = 1 if typename_notebook == 'Sí' else 0
has_ssd = 1 if has_ssd == 'Sí' else 0
processor_brand = get_processor_brand(processor_brand)
so_brand = get_processor_SO(so_brand)

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[ram, typename_gaming, weight, typename_notebook, so_brand, ghz, has_ssd, processor_brand, memory_gb, screen_resolution, inches]],
                    columns=['Ram', 'TypeName_Gaming', 'Weight', 'TypeName_Notebook', 'SO_brand', 'GHz', 'has_SSD', 'Processor_brand', 'Memory_GB', 'ScreenResolution', 'Inches'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    prediction = modelo.predict(input_scaled)

    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} euros')
