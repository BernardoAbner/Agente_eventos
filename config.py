import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "knowledge_base_documents")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "lancedb")
LOG_FILE_PATH = os.path.join(BASE_DIR, "data", "agent.log")
LOG_LEVEL = "INFO" 

LLM_MODEL = "phi3:mini"


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") 

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

VECTOR_DB_TABLE_NAME = "knowledge_base"

CHUNK_SIZE = 700 
CHUNK_OVERLAP = 70 

TOP_K_RESULTS = 3 

ENABLE_OCR = True 

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