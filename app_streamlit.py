import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import json
import warnings
import os

warnings.filterwarnings("ignore")

# CONFIGURACIÓN DE PARÁMETROS FIJOS (Ajustado al RAG de Jupyter)
K_DOCS_FIJO = 6
MAX_TOKENS_FIJO = 256
TEMPERATURE_FIJO = 0.4
TOP_P_FIJO = 0.90
TOP_K_FIJO = 50

# Configuración de página
st.set_page_config(
    page_title="Consultas PNCIL - RAG",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e9f7 100%);
    }
    .doc-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        font-size: 14px;
        color: #1a1a1a;
    }
    .response-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #1a1a1a;
    }
    .header-title {
        text-align: center;
        color: #1976D2;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header-subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Inicialización de modelos (cachear para no recargar)
@st.cache_resource
def load_models():
    """Carga los modelos de embedding y generación"""
    # Configurar rutas si es necesario
    os.environ["HF_HOME"] = "D:/HF/MODEL_CACHE"   
    os.environ["HF_DATASETS_CACHE"] = "D:/HF/MODEL_CACHE"
    
    # Cargar modelo de embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Cargar TinyLlama
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )
    generator = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    return embedding_model, generator

@st.cache_resource
def load_documents_and_index():
    """Carga documentos y crea índice FAISS"""
    # Cargar documentos
    documents = []
    with open(r"C:\Users\agarmendia\Desktop\PNCIL_TinyLlama\pncil_documents.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc['text'])
    
    # Crear embeddings e índice
    embedding_model, _ = load_models()
    doc_embeddings = embedding_model.encode(documents)
    doc_embeddings = np.array(doc_embeddings, dtype='float32')
    
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    
    return index, documents

# Funciones del RAG
def retrieve_documents(query: str, k: int = K_DOCS_FIJO):
    """Recupera los 'k' documentos más relevantes (k=6 fijo)"""
    # Obtener modelos desde cache
    embedding_model, _ = load_models()
    index, documents = load_documents_and_index()
    
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype='float32')
    
    distances, indices = index.search(query_embedding, k)
    
    return [documents[i] for i in indices[0]]

def generate_answer(query: str, retrieved_docs: list):
    """Genera respuesta basada en documentos recuperados"""
    # Obtener generador desde cache
    _, generator = load_models()
    
    context = " ".join(retrieved_docs)
    context_limit = 2000
    if len(context) > context_limit:
        context = context[:context_limit] + "..."
    
    # INSTRUCCIÓN ESTRICTA DEL NOTEBOOK
    system_instruction = "Eres un asistente de preguntas y respuestas cuya ÚNICA fuente de conocimiento es el 'Contexto' proporcionado. Debes responder la 'Pregunta' basandote estrictamente en el contexto, extrae SOLAMENTE esa información. Si el contexto NO PERMITE una respuesta fidedigna, debes responder EXCLUSIVAMENTE: 'No tengo información suficiente para responder a esa pregunta.'."
    
    user_prompt = f"Contexto: {context}\n\nPregunta: {query}"
    prompt = f"<s>[INST] <<SYS>> {system_instruction} <</SYS>> {user_prompt} [/INST]"

    # PARÁMETROS FIJOS DEL NOTEBOOK
    generated_output = generator(
        prompt,
        max_new_tokens=256,
        num_return_sequences=1,
        truncation=True,
        temperature=0.4,   
        top_p=0.9,              
        top_k=50                
    )
    
    answer = generated_output[0]['generated_text']
    
    # LÓGICA DE LIMPIEZA FINAL DEL NOTEBOOK
    if "[/INST]" in answer:
        cleaned_answer = answer.split("[/INST]")[1].strip()
        
        if "<<SYS>>" in cleaned_answer:
            cleaned_answer = cleaned_answer.split("<<SYS>>")[1].strip()
            
        if cleaned_answer.lower().startswith("respuesta:"):
            cleaned_answer = cleaned_answer[len("respuesta:"):].strip()

        if "Pregunta:" in cleaned_answer:
            cleaned_answer = cleaned_answer.split("Pregunta:")[0].strip()
            
        if "</s>" in cleaned_answer:
            cleaned_answer = cleaned_answer.split("</s>")[0].strip()
            
    else:
        cleaned_answer = answer.strip()
    
    return cleaned_answer

# Cargar modelos al inicio
with st.spinner("Cargando modelos y documentos..."):
    load_models()
    load_documents_and_index()
    st.session_state['loaded'] = True

# ============= INTERFAZ STREAMLIT =============

# Header
st.markdown('<div class="header-title">📋 Consultas PNCIL</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Sistema RAG con TinyLlama + MiniLM</div>', unsafe_allow_html=True)
st.markdown("---")

# Main content
query = st.text_area(
    "Tu consulta:",
    value='',
    height=100,
    placeholder="Escribe tu pregunta sobre PNCIL...",
    key="query_input"
)

# Columna para centrar el botón
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    consultar_btn = st.button("🔍 Consultar", type="primary", use_container_width=True)

if consultar_btn and query.strip():
    with st.spinner("🔎 Buscando documentos relevantes..."):
        # Retrieve: K es FIJO a 6
        retrieved_docs = retrieve_documents(query, k=K_DOCS_FIJO) 
        
        if not retrieved_docs:
            st.warning("⚠️ No se encontraron documentos relevantes.")
        else:
            # Generate: Parámetros FIJOS
            with st.spinner("✍️ Generando respuesta..."):
                answer = generate_answer(query, retrieved_docs)
            
            # Mostrar respuesta
            st.markdown("### ✅ Respuesta")
            st.markdown(f'<div class="response-card">{answer}</div>', unsafe_allow_html=True)
            
            # Mostrar documentos recuperados
            st.markdown(f"### 📄 Documentos consultados ({len(retrieved_docs)})")
            
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"📌 Documento {i+1}"):
                    st.markdown(f'<div class="doc-card">{doc}</div>', unsafe_allow_html=True)

elif consultar_btn:
    st.warning("⚠️ Por favor escribe una consulta")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sistema RAG - PNCIL</strong></p>
    <p style='font-size: 12px;'>Powered by TinyLlama 1.1B + SentenceTransformers + FAISS</p>
</div>
""", unsafe_allow_html=True)