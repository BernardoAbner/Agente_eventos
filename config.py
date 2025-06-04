import os
from dotenv import load_dotenv

load_dotenv()

# --- Configurações Gerais ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "knowledge_base_documents")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "lancedb")
LOG_FILE_PATH = os.path.join(BASE_DIR, "data", "agent.log")
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Configurações do LLM (Ollama) ---
# Implementação Primária: Phi-3 Mini
LLM_MODEL = "phi3:mini"
# Implementação Alternativa (Contingência): TinyLlama
# LLM_MODEL = "tinyllama:1.1b-chat-q4_K_M"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") # Garante que Ollama está acessível

# --- Configurações do Modelo de Embedding ---
# Implementação Primária: BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Implementação Alternativa (Contingência): all-MiniLM-L6-v2
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Para FastEmbed (se preferir em vez de sentence-transformers):
# EMBEDDING_MODEL_NAME_FASTEMBED = "BAAI/bge-small-en-v1.5" # FastEmbed usa nomes de SentenceTransformers

# --- Configurações do Banco de Dados Vetorial ---
# Implementação Primária: LanceDB
VECTOR_DB_TABLE_NAME = "knowledge_base"
# Para ChromaDB (Alternativa, se implementado):
# CHROMA_PERSIST_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# --- Configurações de Processamento e RAG ---
# Chunking
CHUNK_SIZE = 700  # Alvo em caracteres (ajuste conforme tokens se tiver tokenizador à parte)
CHUNK_OVERLAP = 70 # Sobreposição em caracteres
# Recuperação (Retrieval)
TOP_K_RESULTS = 3 # Número de chunks relevantes a serem recuperados

# --- Configurações de OCR (Condicional) ---
ENABLE_OCR = True # Mude para True se Tesseract e Pandoc estiverem instalados e aprovados
# Se ENABLE_OCR = True, o sistema tentará usar 'kreuzberg' com Tesseract.
# Para usar modelos OCR pré-baixados com kreuzberg (se Tesseract não for viável):
# OCR_BACKEND = "easyocr" # ou "paddleocr" (requer 'pip install "kreuzberg[easyocr]"')
# Se OCR_BACKEND for definido, certifique-se de que os modelos OCR correspondentes
# sejam baixados para uso offline conforme a documentação de kreuzberg/easyocr/paddleocr.

# --- Template de Prompt para Geração de Resposta ---
PROMPT_TEMPLATE = """Você é um assistente de IA da Empresa X.
Com base APENAS nos seguintes trechos extraídos da base de conhecimento interna, responda à pergunta do usuário.
Seja conciso e direto. Se a informação necessária para responder à pergunta não estiver nos trechos fornecidos, diga explicitamente: 'A informação não foi encontrada na base de conhecimento fornecida.'

Contexto Fornecido:
{contexto_dos_chunks_recuperados}

Pergunta do Usuário:
{pergunta_do_usuario}

Resposta:
"""

# --- Validações Básicas ---
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    print(f"Diretório de documentos criado em: {DOCUMENTS_DIR}")

if not os.path.exists(os.path.join(BASE_DIR, "data")):
    os.makedirs(os.path.join(BASE_DIR, "data"))