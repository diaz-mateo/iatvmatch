import streamlit as st
import os
import subprocess
import json
from gpt4all import GPT4All

# Ruta del modelo
modelo_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

import requests

if not os.path.exists(modelo_path):
    st.write("Descargando el modelo...")
    url = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    response = requests.get(url)
    if response.status_code == 200:
        with open(modelo_path, "wb") as f:
            f.write(response.content)
        st.write("Modelo descargado exitosamente.")
    else:
        st.error("Error al descargar el modelo. Código de estado: " + str(response.status_code))

def cargar_modelo(modelo_path="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
    """
    Función para cargar el modelo GPT4All.
    """
    try:
        st.write("Cargando modelo...")
        modelo = GPT4All(modelo_path)
        st.write("Modelo cargado exitosamente.")
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def generar_recomendaciones(modelo, prompt, max_tokens=512, temperature=0.7):
    """
    Función que utiliza GPT4All para generar recomendaciones a partir de un prompt.
    """
    try:
        with modelo.chat_session():
            respuesta = modelo.generate(prompt, max_tokens=max_tokens, temp=temperature)
        return respuesta
    except Exception as e:
        st.error(f"Error durante la generación: {e}")
        return None

def procesar_respuesta(respuesta_json):
    """
    Función para procesar la respuesta generada en formato JSON y mostrar la información clave.
    """
    try:
        datos = json.loads(respuesta_json)
        recomendaciones = datos.get("recomendaciones", [])
        if not recomendaciones:
            st.write("No se encontraron recomendaciones en la respuesta.")
        else:
            for idx, serie in enumerate(recomendaciones, start=1):
                st.subheader(f"Recomendación {idx}")
                st.markdown(f"**Título:** {serie.get('título', 'Desconocido')}")
                st.markdown(f"**Descripción:** {serie.get('descripción', 'Sin descripción')}")
                st.markdown(f"**Fecha:** {serie.get('fecha', 'No disponible')}")
                elenco = serie.get('elenco', [])
                if isinstance(elenco, list):
                    st.markdown(f"**Elenco:** {', '.join(elenco)}")
                else:
                    st.markdown(f"**Elenco:** {elenco}")
    except json.JSONDecodeError:
        st.write("No se pudo procesar la respuesta como JSON. Respuesta generada:")
        st.write(respuesta_json)

def main():
    st.title("Recomendaciones de Series de TV")
    
    # Cargar el modelo
    modelo = cargar_modelo()
    if modelo is None:
        return

    # Entrada del usuario para el nombre o descripción de la serie
    serie_usuario = st.text_input("Introduce el nombre o descripción de una serie:")

    if st.button("Generar recomendaciones"):
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
            st.subheader("Respuesta completa del modelo:")
            st.code(respuesta, language="json")
            st.write("Procesando la respuesta...")
            procesar_respuesta(respuesta)
        else:
            st.write("No se obtuvo respuesta del modelo.")

if __name__ == "__main__":
    main()
