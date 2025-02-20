import streamlit as st
import urllib.request
import os
import json
from gpt4all import GPT4All

# Definir el nombre del modelo y la URL de descarga
modelo_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
modelo_url = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# Función para verificar y descargar el modelo si no existe
def descargar_modelo():
    if not os.path.exists(modelo_path):
        st.write("Descargando modelo... Esto puede tardar unos minutos.")
        urllib.request.urlretrieve(modelo_url, modelo_path)
        st.success("Modelo descargado exitosamente.")

# Función para cargar el modelo GPT4All
def cargar_modelo():
    try:
        st.write("Cargando modelo...")
        modelo = GPT4All(modelo_path)
        st.success("Modelo cargado exitosamente.")
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para generar recomendaciones con el modelo
def generar_recomendaciones(modelo, prompt, max_tokens=512, temperature=0.7):
    try:
        with modelo.chat_session():
            respuesta = modelo.generate(prompt, max_tokens=max_tokens, temp=temperature)
        return respuesta
    except Exception as e:
        st.error(f"Error durante la generación: {e}")
        return None

# Función para procesar la respuesta generada
def procesar_respuesta(respuesta_json):
    try:
        datos = json.loads(respuesta_json)
        recomendaciones = datos.get("recomendaciones", [])

        if not recomendaciones:
            st.warning("No se encontraron recomendaciones en la respuesta.")
        else:
            for idx, serie in enumerate(recomendaciones, start=1):
                st.subheader(f"Recomendación {idx}:")
                st.write(f"**Título:** {serie.get('título', 'Desconocido')}")
                st.write(f"**Descripción:** {serie.get('descripción', 'Sin descripción')}")
                st.write(f"**Fecha:** {serie.get('fecha', 'No disponible')}")
                elenco = serie.get('elenco', [])
                if isinstance(elenco, list):
                    st.write(f"**Elenco:** {', '.join(elenco)}")
                else:
                    st.write(f"**Elenco:** {elenco}")
    except json.JSONDecodeError:
        st.warning("No se pudo procesar la respuesta como JSON.")
        st.text_area("Respuesta generada:", respuesta_json, height=200)

# Interfaz con Streamlit
st.title("Recomendador de Series de Comedia")

# Descargar el modelo si es necesario
descargar_modelo()

# Cargar el modelo
modelo = cargar_modelo()

# Entrada del usuario
serie_usuario = st.text_input("Introduce el nombre o descripción de una serie:")

if st.button("Generar Recomendaciones"):
    if modelo and serie_usuario.strip():
        prompt = f"""
        Dado el nombre de una serie de TV estadounidense de comedia o sitcom, proporciona una lista de al menos cinco series similares en cuanto a tono, temática y estilo de humor.
        Para cada recomendación, incluye:
            - "título": Nombre de la serie.
            - "descripción": Breve sinopsis de la serie.
            - "fecha": Año de lanzamiento.
            - "elenco": Lista de actores destacados.
        Justifica brevemente por qué cada serie es una buena alternativa, considerando elementos como el reparto, la narrativa y la audiencia objetivo.
        Genera la respuesta en formato JSON con la siguiente estructura:
        {{
            "recomendaciones": [
                {{
                    "título": "Nombre de la serie",
                    "descripción": "Breve sinopsis",
                    "fecha": "Año de lanzamiento",
                    "elenco": ["Actor 1", "Actor 2"]
                }},
                ...
            ]
        }}
        Serie de referencia: "{serie_usuario}"
        """

        st.write("Generando recomendaciones, por favor espere...")
        respuesta = generar_recomendaciones(modelo, prompt)

        if respuesta:
            procesar_respuesta(respuesta)
        else:
            st.error("No se obtuvo respuesta del modelo.")
    else:
        st.warning("Por favor, ingresa una serie válida.")
