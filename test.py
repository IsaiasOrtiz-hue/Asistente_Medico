from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Cargar las variables del archivo .env
load_dotenv()

# Obtener las variables
groq_api_key = os.getenv('GROQ_API_KEY')
model_name = os.getenv('MODEL_NAME')

# Verificar si las variables están bien cargadas
if not groq_api_key or not model_name:
    print("Error: API key o nombre de modelo no están configurados correctamente.")
    exit()

# Inicializar el modelo de chat
try:
    chat = ChatGroq(
        temperature=0,  # Configuración de la temperatura para la generación de respuestas
        groq_api_key=groq_api_key,
        model_name=model_name
    )
    print("Modelo de chat inicializado correctamente.")
except Exception as e:
    print(f"Error al inicializar el modelo: {e}")
