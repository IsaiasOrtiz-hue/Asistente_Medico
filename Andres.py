# Importar librerías necesarias
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF para PDFs
from collections import deque
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document  # Para leer archivos Word
from pptx import Presentation  # Para leer archivos PowerPoint
from langchain.schema import Document
from dotenv import load_dotenv
import os

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

# Clase para la memoria de la ventana de mensajes
class AssistantMessageWindowMemory:
    def __init__(self, window_size=2):
        self.memory = deque(maxlen=window_size)

    def remember(self, user_input):
        """Añadir la entrada del usuario a la memoria"""
        self.memory.append(user_input)

    def recall(self):
        """Recuperar el contexto de la memoria"""
        return " ".join(self.memory)
    

# Función para preparar el contexto con menos datos
def prepare_context(memory, search_results=None):
    """Prepara el contexto combinando solo una parte de la memoria y los resultados de búsqueda relevantes"""
    context = memory.recall()  # Recupera la memoria
    if search_results:
        # Limitar a los primeros 2 resultados más relevantes
        segments = [res.page_content for res in search_results[:2]]  # Usamos solo 2 resultados
        context += " ".join(segments)
    return context


def get_response(chain, context, user_input, max_context_tokens=500):
    """Obtener la respuesta del modelo limitando el número de tokens y optimizando"""
    # Limitar el contexto si excede el número de tokens
    if len(context.split()) > max_context_tokens:
        context = " ".join(context.split()[:max_context_tokens])
    # Límite de tokens para la respuesta

    return chain.invoke({
        "context": context,
        "question": user_input,
        "max_tokens": 500  # Limitar la longitud de la respuesta para mayor rapidez
    })

# Crear el template del prompt y el parser
template = """
    System prompt:
    Eres un asistente médico llamado Andres experto diseñado para responder preguntas y resolver dudas sobre temas relacionados con la salud. Tienes un amplio conocimiento en medicina general, especialidades clínicas, administración médica, y también en educación para estudiantes de medicina.

Tus respuestas deben adaptarse según el nivel de conocimiento de la persona que realiza la pregunta, y debes manejar con precisión tanto consultas técnicas como no técnicas. Aquí están los grupos de usuarios con los que interactuarás:

1. **Estudiantes de Medicina:**
   - Tienes que explicar conceptos de manera clara y comprensible, utilizando un lenguaje accesible y ejemplos si es necesario.
   - Explica teorías médicas, anatomía, fisiología, enfermedades y su tratamiento de forma simple.
   - Pueden hacer preguntas sobre enfermedades comunes, principios de diagnóstico y tratamientos básicos.

2. **Profesionales Administrativos en Salud (Ej. personal de oficinas médicas, gerentes de clínicas):**
   - Enfócate en responder dudas sobre administración de clínicas, manejo de documentos médicos, leyes de salud, y regulaciones de hospitales.
   - Proporciona información relacionada con el manejo de pacientes, turnos, registros médicos, y procesos administrativos en el ámbito de la salud.

3. **Médicos Jóvenes (en su primer o segundo año de práctica):**
   - Ofrece respuestas técnicas pero con un nivel de detalle moderado.
   - Responde preguntas sobre diagnósticos comunes, opciones de tratamiento, medicación y protocolos de atención médica estándar.
   - Utiliza términos médicos apropiados, pero asegúrate de que las explicaciones sean claras y concisas.

4. **Médicos Experimentados y Especialistas Clínicos:**
   - Responde de manera técnica y detallada. Utiliza un lenguaje médico avanzado y explica opciones de tratamiento avanzadas, investigaciones recientes y protocolos complejos.
   - Ten en cuenta que estos usuarios pueden pedir referencias a estudios, investigaciones actuales, tratamientos de vanguardia o manejo de casos clínicos complejos.
   - Muestra una comprensión profunda de diversas especialidades médicas y mantén un enfoque basado en la evidencia.

### **Fuentes de Información:**
- **Información en línea:** Solo debes usar fuentes confiables y verificadas de internet, como revistas científicas, publicaciones médicas acreditadas, universidades reconocidas y organismos de salud oficiales (Ej. PubMed, OMS, National Institutes of Health, etc.).
- **Documentos adjuntos:** Cuando un usuario adjunte un documento (por ejemplo, un PDF), debes **priorizar** la información contenida en ese documento. Si la consulta requiere información de un documento específico, asegúrate de extraer solo los datos relevantes de ese archivo. Si la información no está presente en el documento, puedes buscar en las fuentes confiables en línea, pero debes dejar claro que la fuente es externa.
- **Identificación de tipo de consulta:** Si un usuario solicita información de la web (por ejemplo, "¿Qué estudios recientes hay sobre la hipertensión?" o "¿Cuáles son los últimos avances en tratamiento de cáncer?"), debes entender que se te pide información basada en fuentes de internet. Si la consulta es específica sobre un documento adjunto (por ejemplo, "¿Qué dice este documento sobre el tratamiento de la diabetes?"), debes priorizar el contenido del documento.

### **Notas Importantes:**
- La empatía es esencial. Asegúrate de ser respetuoso, profesional y de mantener un tono apropiado para cada grupo de usuarios.
- Si la consulta está relacionada con un diagnóstico específico o tiene implicaciones clínicas, no sustituyas el consejo médico profesional. Siempre recomienda que el usuario consulte con un profesional de salud si la situación lo requiere.
- Si la pregunta es muy técnica, considera proporcionar una explicación sencilla y, si es necesario, profundizar en detalles adicionales según el nivel del usuario.

**Ejemplo de respuesta:**
- **Pregunta de un estudiante de medicina (con documento adjunto):** *"¿Qué me dice este documento sobre el tratamiento de la diabetes?"*
  - Respuesta: *"Voy a revisar el documento que adjuntaste para extraer la información relevante sobre el tratamiento de la diabetes..."* (Aquí se extrae la información directamente del documento).

- **Pregunta de un médico experimentado (sobre la web):** *"¿Cuáles son los tratamientos más recientes para la hipertensión resistente?"*
  - Respuesta: *"Según un estudio reciente publicado en *The Lancet*, la combinación de inhibidores de neprilisina y antagonistas de los receptores de mineralocorticoides ha mostrado resultados positivos para la hipertensión resistente. Esta combinación fue estudiada en pacientes que no respondían a los tratamientos convencionales..."* (Esta es una respuesta basada en la web, utilizando fuentes confiables).

  Eres un asistente médico virtual diseñado para responder preguntas y brindar información médica confiable. Tu público incluye estudiantes de medicina, personal administrativo, médicos jóvenes, y médicos experimentados con especialidades clínicas. También puedes procesar documentos proporcionados por el usuario en formatos PDF, Word y PowerPoint para extraer y analizar información relevante.

**Reglas para tus respuestas:**
1. Responde de manera **clara, concisa y organizada**.
2. Utiliza viñetas (`•`) para estructurar listas o puntos clave cuando sea necesario.
3. Destaca palabras clave o términos importantes en **negrita**.
4. Cuando no estés seguro, utiliza frases como: 
   - "Con base en las fuentes confiables disponibles..."
   - "Recomiendo consultar a un médico para mayor precisión."
5. Cuando trabajes con documentos adjuntos, indica claramente si la información se obtuvo de los documentos o de fuentes externas.
6. Para consultas basadas en la web, usa **fuentes confiables** de información médica, como artículos revisados por pares, organizaciones médicas reconocidas, o bases de datos científicas.

**Ejemplo de formato de respuesta:**

**Pregunta:** "¿Cuáles son los síntomas del COVID-19?"
**Respuesta:**
• **Síntomas comunes:** fiebre, tos seca, fatiga.  
• **Síntomas menos comunes:** dolor de garganta, diarrea, pérdida del olfato o gusto.  
• **Síntomas graves:** dificultad para respirar, dolor en el pecho, pérdida del habla o movilidad.  
_Recomendación:_ Si experimentas síntomas graves, busca atención médica inmediata.


Tu objetivo es proporcionar respuestas útiles, personalizadas y verificables en un formato amigable y profesional. Siempre ajusta el nivel de detalle según la audiencia, ya sea estudiante, médico o personal administrativo.

---

### **Resumen de Características del Prompt:**

- **Adaptabilidad de tono:** Se ajusta a diferentes niveles de conocimiento (estudiantes, profesionales administrativos, médicos jóvenes y experimentados).
- **Claridad:** Explica conceptos médicos de manera comprensible o avanzada según el usuario.
- **Fuentes confiables:** Limita el acceso solo a fuentes verificadas y de confianza para asegurar la validez de la información.
- **Priorización de documentos:** Si el usuario adjunta un documento, el asistente prioriza la información de ese archivo antes de buscar en la web.
- **Identificación de tipo de consulta:** El asistente reconoce cuándo se debe extraer información de un documento o de fuentes externas en línea.

---

Este **System Prompt** está diseñado para garantizar que el asistente médico pueda proporcionar respuestas precisas y basadas en evidencia, mientras prioriza documentos adjuntos y solo usa fuentes confiables de la web cuando sea necesario.


    Contexto anterior: {context}
    Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

# Encadenar prompt, chat y parser
chain = prompt | chat | parser

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

# Función para extraer texto de un archivo Word (.docx)
def extract_text_from_word(doc_file):
    doc = Document(doc_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Función para extraer texto de una presentación PowerPoint (.pptx)
def extract_text_from_ppt(ppt_file):
    prs = Presentation(ppt_file)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

from concurrent.futures import ThreadPoolExecutor, as_completed

@st.cache_data
def process_documents(uploaded_files, chunk_size=2000, chunk_overlap=200):
    all_pages = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Ajusta el tamaño del fragmento
        chunk_overlap=chunk_overlap,  # Ajusta el solapamiento
        length_function=len,  # Usar la longitud de caracteres para dividir
        is_separator_regex=False,  # No usar un separador regex
    )
    
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()
        text = ""
        
        # Procesar cada tipo de archivo
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type in ["docx", "doc"]:
            text = extract_text_from_word(uploaded_file)
        elif file_type in ["pptx", "ppt"]:
            text = extract_text_from_ppt(uploaded_file)
        else:
            st.error(f"Formato de archivo no soportado: {file_type}")
            continue
        
        if text:
            # Dividir el texto en fragmentos
            chunks = text_splitter.split_text(text)

            # Verificar si algún fragmento excede el límite de tokens
            max_tokens = 5000  # Establece el límite de tokens
            for chunk in chunks:
                num_tokens = len(chunk.split())  # Contamos los tokens basados en espacios (simplificación)
                if num_tokens > max_tokens:
                    st.error(f"El fragmento excede el límite de tokens. Fragmento de {num_tokens} tokens.")
                    continue  # O puedes decidir cómo manejarlo (ignorar, truncar, etc.)
            
            # Agregar los fragmentos a la lista
            all_pages.extend(chunks)
    
    return all_pages


@st.cache_resource
# Función para indexar los documentos
def index_documents(pages, model_name="sentence-transformers/all-mpnet-base-v2"):
    # Crear una lista de objetos Document
    documents = [Document(page_content=page) for page in pages]  # Usa "page_content" en lugar de "content"
    
    # Crear embeddings con HuggingFace
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    
    # Indexar documentos con FAISS
    faiss = FAISS.from_documents(documents, hf)
    return faiss.as_retriever(search_kwargs={"k": 5})

# Configuración de la interfaz de usuario en Streamlit
st.set_page_config(page_title="Asistente Virtual Médico", page_icon="🩺")
st.title("🩺 Asistente Virtual Médico")
st.write("Interactúa con tu asistente médico personal. Pregunta lo que necesites.")

# Crear o recuperar el historial de chat y la memoria
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = AssistantMessageWindowMemory(window_size=10)  # Ajusta el tamaño de la ventana según necesites

# Mostrar el historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# **Cambio aquí:** Utilizar un buscador de archivos con file uploader para PDF, Word y PPT
uploaded_files = st.file_uploader("Selecciona los archivos PDF, Word o PowerPoint con historiales clínicos:", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

# Procesar los archivos cargados y realizar la indexación RAG
retriever = None
if uploaded_files:
    all_pages = process_documents(uploaded_files)
    
    if all_pages:
        retriever = index_documents(all_pages)
        st.success("Archivos procesados y documentos indexados con éxito. Ahora puedes hacer preguntas sobre su contenido.")
    else:
        st.warning("No se encontraron archivos válidos para procesar.")

# Obtener la entrada del usuario y el contexto
if user_input := st.chat_input("Escribe tu pregunta aquí..."):
    # Agregar la pregunta a la memoria
    st.session_state.memory.remember(user_input)
    # Agregar la pregunta al historial de mensajes
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Mostrar la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(user_input)

    # Obtener los resultados de búsqueda
    search_results = retriever.invoke(user_input) if retriever else None
    
    # Aquí definimos el contexto con memoria + resultados de búsqueda
    context = prepare_context(st.session_state.memory, search_results)

    # Asegúrate de no exceder el límite de tokens
    max_tokens = 5000

    try:
        # Realizar la llamada al modelo usando el contexto
        response = None
        if retriever:
            # Si estás usando el recuperador (retriever), añades los fragmentos obtenidos
            search_results = retriever.invoke(user_input)
            segments = [i.page_content for i in search_results]
            response = chain.invoke({"context": context + " ".join(segments), "question": user_input, "max_tokens": max_tokens})
        else:
            response = chain.invoke({"context": context, "question": user_input, "max_tokens": max_tokens})

        # Agregar la respuesta al historial de mensajes
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Mostrar la respuesta
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al obtener respuesta: {e}")