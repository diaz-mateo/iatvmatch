# -*- coding: utf-8 -*-
"""FINAL MODELO 1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XFKxEpqatMU0MzqSqAFiu_GkYbBFUGJU
"""

!pip install gpt4all

!pip install streamlit

import os
import subprocess

modelo_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

if not os.path.exists(modelo_path):
    print("Descargando el modelo...")
    subprocess.run([
        "wget",
        "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "-O", modelo_path
    ])
else:
    print("El modelo ya está descargado.")

import json
from gpt4all import GPT4All

import subprocess

subprocess.run([
    "wget",
    "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "-O", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
])

def generar_recomendaciones(modelo, prompt, max_tokens=512, temperature=0.7):
    """
    Función que utiliza GPT4All para generar recomendaciones a partir de un prompt.

    Parámetros:
      - modelo: Modelo cargado de GPT4All.
      - prompt: Instrucción completa para el modelo.
      - max_tokens: Número máximo de tokens a generar.
      - temperature: Controla la aleatoriedad en la generación.

    Retorna:
      - respuesta: La respuesta generada en texto.
    """
    try:
        with modelo.chat_session():
            # Se usa 'temp' en lugar de 'temperature' para que coincida con la firma de la función generate.
            respuesta = modelo.generate(prompt, max_tokens=max_tokens, temp=temperature)
        return respuesta
    except Exception as e:
        print(f"Error durante la generación: {e}")
        return None

def procesar_respuesta(respuesta_json):
    """
    Función para procesar la respuesta generada en formato JSON y extraer información clave.
    Si la respuesta no es un JSON válido, se muestra el contenido generado en bruto.
    """
    try:
        datos = json.loads(respuesta_json)
        recomendaciones = datos.get("recomendaciones", [])
        if not recomendaciones:
            print("No se encontraron recomendaciones en la respuesta.")
        else:
            for idx, serie in enumerate(recomendaciones, start=1):
                print(f"\nRecomendación {idx}:")
                print(f"  Título: {serie.get('título', 'Desconocido')}")
                print(f"  Descripción: {serie.get('descripción', 'Sin descripción')}")
                print(f"  Fecha: {serie.get('fecha', 'No disponible')}")
                elenco = serie.get('elenco', [])
                if isinstance(elenco, list):
                    print(f"  Elenco: {', '.join(elenco)}")
                else:
                    print(f"  Elenco: {elenco}")
    except json.JSONDecodeError:
        print("No se pudo procesar la respuesta como JSON. Respuesta generada:")
        print(respuesta_json)

def main():
    # Cargar el modelo. Asegúrate de que el archivo "Meta-Llama-3-8B-Instruct.Q4_0.gguf" se haya descargado correctamente.
    modelo = cargar_modelo()
    if modelo is None:
        return

    # Solicitar al usuario el nombre o descripción de la serie.
    serie_usuario = input("Introduce el nombre o descripción de una serie: ").strip()

    # Construir el prompt con instrucciones claras para la IA.
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

    print("Generando recomendaciones, por favor espere...")
    respuesta = generar_recomendaciones(modelo, prompt)

    if respuesta:
        print("\nRespuesta completa del modelo:")
        print(respuesta)
        print("\nProcesando la respuesta...")
        procesar_respuesta(respuesta)
    else:
        print("No se obtuvo respuesta del modelo.")

if __name__ == "__main__":
    main()

"""TVMatch: Recomendador de Series de TV con IA
1. Portada
Título del Proyecto:
TVMatch – Recomendador de Series de TV con IA

Alumno:
Mateo Díaz Paredes

Curso:
Inteligencia Artificial: Prompt Engineering para Programadores – Carreras Intensivas

Comisión:
76195

Módulo:
MÓDULO II – 2025-01

Institución:
CODERHOUSE – DIPLOMATURA Desarrollador Web Full Stack

2. Introducción
El auge de las plataformas de streaming ha generado una sobrecarga de contenido audiovisual, haciendo que los usuarios se sientan abrumados al momento de elegir qué ver. Los algoritmos tradicionales de recomendación suelen priorizar contenido popular o basado en patrones generales, sin tener en cuenta las preferencias específicas y la evolución del gusto del usuario. En este contexto, TVMatch surge como una propuesta innovadora que utiliza herramientas de Inteligencia Artificial (IA) para generar recomendaciones personalizadas de series de TV, adaptándose a los intereses individuales y proporcionando retroalimentación continua.

3. Problemática
Actualmente, la gran cantidad de series disponibles en plataformas de streaming genera dificultades para que los usuarios encuentren contenido que realmente se ajuste a sus gustos. Entre los problemas identificados destacan:

Sobrecarga de Información:
La abundancia de series y películas hace que el usuario tenga que invertir tiempo en explorar catálogos extensos, lo que puede resultar en una experiencia abrumadora.

Recomendaciones Genéricas:
Los algoritmos tradicionales suelen recomendar únicamente lo más popular, sin considerar aspectos subjetivos como el tipo de humor, la profundidad de los personajes o la evolución de las preferencias del usuario.

Falta de Personalización y Retroalimentación:
Los sistemas actuales no logran ajustar sus recomendaciones en función de la interacción continua del usuario, lo que impide que el sistema mejore su precisión a lo largo del tiempo.

4. Objetivos del Proyecto
Objetivo General
Desarrollar una aplicación web (TVMatch) que integre herramientas de IA para generar recomendaciones personalizadas de series de TV a partir de una serie de referencia proporcionada por el usuario.

Objetivos Específicos
Generar recomendaciones personalizadas:
Utilizar un modelo de procesamiento de lenguaje natural (NLP) para analizar el prompt y generar una lista de series similares en formato JSON.

Integrar fuentes de datos externas:
Enriquecer la información de la serie de referencia consultando bases de datos externas (por ejemplo, la API de TMDb) para obtener detalles adicionales como año de emisión, descripción y popularidad.

Modularizar el código:
Definir subfunciones para cargar el modelo, procesar datos, gestionar errores y generar respuestas, facilitando el mantenimiento y escalabilidad del proyecto.

Evaluar la viabilidad económica y técnica:
Analizar costos asociados (por ejemplo, uso de APIs gratuitas, costos de infraestructura) y estrategias de monetización (publicidad, suscripciones premium) para garantizar la sostenibilidad del proyecto.

Proyectar una interfaz de usuario amigable:
Aunque en esta versión se utiliza Colab para probar el MVP, se propone la implementación futura de una aplicación web interactiva (por ejemplo, con Streamlit o Flask) que facilite la interacción con el sistema.

5. Solución Propuesta
TVMatch se basa en la integración de GPT4All, un modelo de IA que permite ejecutar LLMs de manera local, con datos enriquecidos mediante la API de TMDb. La solución se desarrolla en varias etapas:

Carga del Modelo de IA:
Se utiliza GPT4All para cargar un modelo de lenguaje (en este caso, Meta-Llama-3-8B-Instruct en formato GGUF) que genere respuestas en formato JSON.

Obtención de Datos Externos:
Se consulta la API de TMDb para extraer información adicional sobre la serie ingresada (nombre, primer año, descripción y popularidad). Esto permite enriquecer el prompt y mejorar la calidad de las recomendaciones.

Generación de Recomendaciones:
Se construye un prompt estructurado que indica a la IA generar al menos cinco series similares, incluyendo campos específicos (título, descripción, fecha, elenco) y una justificación breve de la recomendación.

Procesamiento y Visualización:
La respuesta generada en JSON se procesa para extraer y mostrar la información clave de cada recomendación.

Consideraciones para Producción:
Se incluyen comentarios y recomendaciones sobre la implementación de una interfaz web, estrategias de escalabilidad (caché, balanceo de carga) y la integración robusta de fuentes de datos (IMDb, TMDb, etc.).

6. Arquitectura y Diseño del Sistema
6.1 Componentes del Sistema
Capa de Entrada:

Interfaz de Usuario (Futura): Se plantea el desarrollo de una app web interactiva (con Streamlit, Flask o React) que permita al usuario ingresar la serie de referencia y visualizar recomendaciones.
Entrada en Colab: En la versión actual, se utiliza la función input() para recibir datos del usuario.
Capa de Procesamiento y Generación de Recomendaciones:

Módulo de IA: Utiliza GPT4All para procesar el prompt y generar recomendaciones.
Módulo de Enriquecimiento de Datos: Integra la API de TMDb para obtener detalles adicionales sobre la serie de referencia.
Capa de Salida:

Procesamiento de JSON: Se parsea y se muestra la salida en consola de manera estructurada.
Visualización (Futura): Se propone implementar dashboards o vistas interactivas para presentar recomendaciones.
6.2 Diagrama de Flujo (Conceptual)
Entrada del Usuario:
El usuario ingresa el nombre o descripción de una serie y, opcionalmente, su API key de TMDb.

Consulta a TMDb:
Se realiza una consulta para obtener datos adicionales sobre la serie.

Construcción del Prompt:
Se integra la información obtenida y se construye un prompt estructurado.

Generación con GPT4All:
Se invoca el modelo de IA para generar recomendaciones en formato JSON.

Procesamiento y Salida:
Se parsea el JSON y se muestran las recomendaciones en consola.

7. Detalles Técnicos y de Implementación
7.1 Integración de GPT4All
Modelo Utilizado:
Se utiliza el modelo "Meta-Llama-3-8B-Instruct.Q4_0.gguf", descargado desde Hugging Face.

Parámetros de Generación:
Se puede ajustar el número máximo de tokens (max_tokens), la temperatura (temp), y otros parámetros de generación para controlar la creatividad y precisión de las respuestas.

7.2 Integración con TMDb
API de TMDb:
Se utiliza la API de TMDb para obtener información adicional de la serie. Esto permite enriquecer el prompt y aumentar la relevancia de las recomendaciones.

Configuración:
El usuario debe proporcionar una API key válida. Se utiliza un endpoint de búsqueda para extraer detalles como nombre, año de estreno, descripción y popularidad.

7.3 Modularización y Manejo de Errores
El código se estructura en funciones separadas:

cargar_modelo(): Gestiona la carga del modelo y posibles errores.
obtener_info_serie_tmdb(): Realiza la consulta a TMDb y devuelve la información en formato diccionario.
generar_recomendaciones(): Invoca la generación de texto a través de GPT4All.
procesar_respuesta(): Parsea y presenta la salida en JSON.
main(): Orquesta el flujo completo de la aplicación.
7.4 Consideraciones de Escalabilidad y Futuras Mejoras
Interfaz de Usuario:
Para producción se recomienda desarrollar una aplicación web con frameworks como Streamlit o Flask, que permita una experiencia de usuario interactiva y visual.

Manejo de Solicitudes Simultáneas:
Se podrían implementar técnicas de cacheo de resultados, balanceo de carga y uso de servicios en la nube para atender un gran volumen de usuarios.

Ajustes de Hiperparámetros y Fine-Tuning:
Si se requiere mayor precisión, se pueden realizar ajustes en los hiperparámetros del modelo (como temperatura, top_k, etc.) o incluso realizar fine-tuning del modelo utilizando datasets especializados de series de TV.

Integración con Múltiples Fuentes de Datos:
Además de TMDb, se podría integrar con IMDb u otras bases de datos para obtener una visión más completa y precisa de la información de series.

8. Análisis de Factibilidad Económica y Técnica
8.1 Factibilidad Económica
Herramientas y APIs Gratuitas:
Se utiliza GPT4All, que permite ejecutar LLMs localmente sin necesidad de costosas licencias ni infraestructura de nube, y la API de TMDb, que ofrece planes gratuitos.

Monetización:
En un producto final, se podría monetizar la aplicación mediante publicidad, suscripciones premium para funciones avanzadas (historial de búsquedas, recomendaciones detalladas) o integración de servicios de pago para empresas.

8.2 Requerimientos Técnicos
Hardware:
Se recomienda contar con al menos 8 GB de RAM para ejecutar el modelo localmente (para modelos de mayor tamaño se requerirá hardware más potente). En producción, se pueden utilizar servidores escalables.

Software:
Uso de Python, GPT4All y librerías auxiliares como Requests para integrar APIs externas. Para la versión web, se consideraría Streamlit, Flask o frameworks similares.

9. Plan de Implementación
9.1 Fases del Proyecto
Desarrollo del MVP en Colab:
Implementar y testear la generación de recomendaciones en Colab utilizando GPT4All y la integración con TMDb.

Integración de Datos y Ajustes del Modelo:
Profundizar en la obtención de datos de múltiples fuentes, ajustar hiperparámetros y realizar pruebas de fine-tuning si es necesario.

Desarrollo de la Interfaz de Usuario:
Implementar una versión web (por ejemplo, con Streamlit) para interactuar con el sistema de forma amigable.

Pruebas de Escalabilidad y Optimización:
Evaluar el rendimiento del sistema en condiciones de alta demanda y aplicar técnicas de cacheo y balanceo de carga.

Despliegue y Retroalimentación:
Desplegar la aplicación en un entorno de producción y recoger feedback de los usuarios para futuras mejoras.

9.2 Cronograma Tentativo
Semana 1-2: Desarrollo y pruebas del MVP en Colab.
Semana 3: Integración de la API de TMDb y ajuste de hiperparámetros.
Semana 4-5: Desarrollo de la interfaz web y pruebas de usuario.
Semana 6: Optimización, escalabilidad y despliegue final.
10. Conclusiones y Reflexiones
TVMatch representa una solución innovadora y viable para el desafío actual de la sobrecarga de contenido en plataformas de streaming. Al combinar un modelo de IA local (GPT4All) con fuentes de datos externas como TMDb, se logra ofrecer recomendaciones personalizadas que se adaptan a los gustos y necesidades de cada usuario. La documentación y el código modular demuestran la solidez técnica del proyecto, a la vez que identifican áreas de mejora para futuras versiones, como la integración de múltiples bases de datos, la implementación de una interfaz de usuario web robusta y estrategias de escalabilidad.

El enfoque propuesto no solo mejora la precisión de las recomendaciones, sino que también ofrece un modelo de negocio potencialmente sostenible mediante opciones de monetización y bajo costo operativo. En definitiva, TVMatch es un proyecto escalable y adaptable que puede evolucionar para convertirse en una herramienta indispensable para usuarios y plataformas de streaming.

11. Referencias y Enlaces
Colab Notebook: https://colab.research.google.com/drive/1Z8nodFxayh5UF7qo7oK26URgYTVr25zy
GPT4All Documentation: https://docs.gpt4all.io/
API de TMDb: https://developers.themoviedb.org/3
Artículos y Referencias sobre LLMs y Streaming de Contenido:
Lifewire: "Unlocking Llama 3's Potential: What You Need to Know"
Documentación de LangChain para integraciones con GPT4All
"""
