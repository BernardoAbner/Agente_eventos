import os
import csv
from pypdf import PdfReader
from docx import Document as DocxDocument
from config import ENABLE_OCR 
from utils import app_logger

if ENABLE_OCR:
    try:
        from kreuzberg import Kreis, get_parser
        app_logger.info("Modo OCR habilitado. 'kreuzberg' importado com sucesso.")

    except ImportError:
        app_logger.warning("'kreuzberg' não encontrado. OCR não estará disponível.")
        ENABLE_OCR = False
    except Exception as e:
        app_logger.error(f"Erro ao configurar kreuzberg para OCR: {e}")
        ENABLE_OCR = False


def extract_text_from_txt(file_path: str) -> str:
    """Extrai texto de arquivos TXT."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        app_logger.error(f"Erro ao ler arquivo TXT {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Extrai texto de arquivos PDF (baseados em texto)."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text and ENABLE_OCR:
            app_logger.info(f"PDF {file_path} não contém texto extraível. Tentando OCR...")
            try:
                parser = get_parser('ocr') 
                kreis = Kreis(parser=parser)
                parsed_doc = kreis.parse_file(file_path)
                return parsed_doc.text_content if parsed_doc else ""
            except Exception as e_ocr:
                app_logger.error(f"Falha no OCR para PDF {file_path}: {e_ocr}")
                return ""
        return text
    except Exception as e:
        app_logger.error(f"Erro ao processar PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extrai texto de arquivos DOCX."""
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        app_logger.error(f"Erro ao processar DOCX {file_path}: {e}")
        return ""

def extract_text_from_csv(file_path: str) -> str:
    """Extrai texto de arquivos CSV, tratando cada linha como um parágrafo."""
    try:
        text_content = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                text_content.append(", ".join(header))

            for row in reader:
                row_text = ", ".join([val if val else "N/A" for val in row])
                text_content.append(row_text)
        return "\n".join(text_content)
    except Exception as e:
        app_logger.error(f"Erro ao processar CSV {file_path}: {e}")
        return ""

def load_documents_from_directory(directory_path: str) -> list[dict]:
    """
    Carrega e processa todos os documentos suportados de um diretório.
    Retorna uma lista de dicionários, cada um contendo 'source' (nome do arquivo) e 'content'.
    """
    processed_documents = []
    supported_extensions = {
        ".txt": extract_text_from_txt,
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".csv": extract_text_from_csv,
    }

    if ENABLE_OCR:
        app_logger.info("Verificando dependências de OCR (Tesseract, Pandoc)...")

        pass


    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in supported_extensions:
                app_logger.info(f"Processando arquivo: {filename}...")
                try:
                    content = supported_extensions[ext](file_path)
                    if content and content.strip():
                        processed_documents.append({"source": filename, "content": content.strip()})
                        app_logger.debug(f"Conteúdo extraído de {filename} (primeiros 100 chars): {content[:100].strip()}...")
                    else:
                        app_logger.warning(f"Nenhum conteúdo extraído ou conteúdo vazio para {filename}.")
                except Exception as e:
                    app_logger.error(f"Falha ao processar o arquivo {filename}: {e}")
            elif ENABLE_OCR and ext not in ['.txt', '.csv']:
                app_logger.info(f"Tentando OCR para arquivo não textual: {filename} (ext: {ext})")
                try:
                    parser = get_parser('ocr') 
                    kreis = Kreis(parser=parser)
                    parsed_doc = kreis.parse_file(file_path)
                    if parsed_doc and parsed_doc.text_content:
                        content = parsed_doc.text_content.strip()
                        processed_documents.append({"source": filename, "content": content})
                        app_logger.debug(f"Conteúdo OCR de {filename} (primeiros 100 chars): {content[:100].strip()}...")
                    else:
                         app_logger.warning(f"Nenhum conteúdo OCR extraído de {filename}.")
                except Exception as e_ocr_generic:
                    app_logger.error(f"Falha no OCR para arquivo genérico {filename}: {e_ocr_generic}")
        


    app_logger.info(f"Total de {len(processed_documents)} documentos processados com sucesso.")
    return processed_documents