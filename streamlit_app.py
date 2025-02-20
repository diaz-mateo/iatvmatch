import json
import streamlit as st
from gpt4all import GPT4All

def cargar_modelo(modelo_path="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
    """
    Función para cargar el modelo GPT4All.
    """
    try:
        modelo = GPT4All(modelo_path)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def generar_recomendaciones(modelo, prompt, max_tokens=512, temperature=0.7):
    """
    Genera recomendaciones basadas en el prompt.
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
    Procesa la respuesta generada en formato JSON.
    """
    try:
        datos = json.loads(respuesta_json)
        return datos.get("recomendaciones", [])
    except json.JSONDecodeError:
        return None

def main():
    st.title("Recomendador de Series con GPT4All")
    
    modelo = cargar_modelo()
    if modelo is None:
        return
    
    serie_usuario = st.text_input("Introduce el nombre o descripción de una serie:")
    
    if st.button("Generar recomendaciones"):
        if serie_usuario:
            prompt = f"""
            Dado el nombre de una serie de TV estadounidense de comedia o sitcom, proporciona una lista de al menos cinco series similares en cuanto a tono, temática y estilo de humor.
            Para cada recomendación, incluye:
                - "título": Nombre de la serie.
                - "descripción": Breve sinopsis de la serie.
                - "fecha": Año de lanzamiento.
                - "elenco": Lista de actores destacados.
            Justifica brevemente por qué cada serie es una buena alternativa.
            Responde en formato JSON con la estructura:
            {{
                "recomendaciones": [
                    {{
                        "título": "Nombre de la serie",
                        "descripción": "Breve sinopsis",
                        "fecha": "Año de lanzamiento",
                        "elenco": ["Actor 1", "Actor 2"]
                    }}
                ]
            }}
            Serie de referencia: "{serie_usuario}"
            """
            
            st.write("Generando recomendaciones...")
            respuesta = generar_recomendaciones(modelo, prompt)
            
            if respuesta:
                recomendaciones = procesar_respuesta(respuesta)
                if recomendaciones:
                    for idx, serie in enumerate(recomendaciones, start=1):
                        st.subheader(f"Recomendación {idx}:")
                        st.write(f"**Título:** {serie.get('título', 'Desconocido')}")
                        st.write(f"**Descripción:** {serie.get('descripción', 'Sin descripción')}")
                        st.write(f"**Fecha:** {serie.get('fecha', 'No disponible')}")
                        elenco = ", ".join(serie.get('elenco', [])) if isinstance(serie.get('elenco'), list) else serie.get('elenco', '')
                        st.write(f"**Elenco:** {elenco}")
                else:
                    st.write("No se pudo procesar la respuesta en formato JSON.")
            else:
                st.write("No se obtuvo respuesta del modelo.")

if __name__ == "__main__":
    main()
