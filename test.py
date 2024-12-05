# =======================
# 1. Importación de Librerías
# =======================
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import AssistantMessageWindowMemory
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
from langchain_groq import ChatGroq
# =======================
# 2. Configuración Inicial
# =======================
load_dotenv()  # Carga las variables del entorno

# Variables de entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "groq")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))

# Inicialización del modelo y memoria
model = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
memory = AssistantMessageWindowMemory(k=5)

# Inicialización de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# =======================
# 3. Funciones Auxiliares
# =======================

# 3.1 Procesamiento de archivos
def extract_text_from_pdf(file):
    """Extrae texto de un archivo PDF."""
    reader = PdfReader(file)
    text = " ".join(page.extract_text() for page in reader.pages)
    return text

def extract_text_from_word(file):
    """Extrae texto de un archivo Word (.docx)."""
    document = Document(file)
    text = " ".join(paragraph.text for paragraph in document.paragraphs)
    return text

def extract_text_from_ppt(file):
    """Extrae texto de un archivo PowerPoint (.pptx)."""
    presentation = Presentation(file)
    text = " ".join(shape.text for slide in presentation.slides for shape in slide.shapes if shape.has_text_frame)
    return text

# 3.2 Preparación de contexto
def prepare_context(memory, search_results):
    """Combina memoria y resultados de búsqueda para generar un contexto relevante."""
    memory_content = memory.load_memory_variables({})
    past_messages = memory_content.get("history", [])
    context = "".join(msg.content for msg in past_messages)
    context += "\n".join(search_results)
    return context

# 3.3 Indexación de documentos
def index_documents(pages):
    """Crea un índice FAISS con los embeddings de los fragmentos procesados."""
    vector_store = FAISS.from_texts(pages, embedding_model)
    return vector_store.as_retriever()

# =======================
# 4. Interfaz del Usuario
# =======================
st.title("Asistente Virtual Médico con IA")

# Carga de archivos
uploaded_files = st.file_uploader("Sube tus documentos aquí:", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
all_pages = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            all_pages.append(extract_text_from_pdf(file))
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            all_pages.append(extract_text_from_word(file))
        elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            all_pages.append(extract_text_from_ppt(file))
    st.success("Documentos procesados correctamente.")

# Indexación (si hay documentos)
if all_pages:
    retriever = index_documents(all_pages)
    st.success("Documentos indexados con éxito.")

# Chat interactivo
st.subheader("Haz tu consulta")
user_input = st.chat_input("Escribe tu pregunta aquí:")

# Manejo del chat
if user_input:
    # Recupera contexto
    search_results = retriever.get_relevant_documents(user_input) if all_pages else []
    context = prepare_context(memory, search_results)

    # Genera respuesta
    chain = ConversationalRetrievalChain(llm=model, retriever=retriever, memory=memory)
    response = chain.invoke({"question": user_input, "context": context})

    # Muestra la respuesta
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response["output_text"])
    st.session_state.messages.append({"user": user_input, "assistant": response["output_text"]})
