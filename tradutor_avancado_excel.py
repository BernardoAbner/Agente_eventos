# tradutor_avancado_excel.py
import pandas as pd
import ollama
import argparse
import os
from tqdm import tqdm

# Importa as configurações do LLM do seu arquivo config.py
from config import LLM_MODEL, OLLAMA_HOST
from utils import app_logger

# Define quantos dados da planilha serão enviados ao LLM de cada vez.
# Ajuste conforme o tamanho da sua planilha e a memória disponível.
# Se receber erros de contexto, diminua este número.
ROWS_PER_CHUNK = 30 

PROMPT_TEMPLATE_AVANCADO = """Você é um assistente de análise de dados. Sua tarefa é analisar os dados de uma planilha, apresentados abaixo em formato CSV, e descrevê-los em um texto narrativo e explicativo em português.

**Instruções:**
1.  Não apenas liste as linhas. Crie parágrafos que expliquem o que os dados significam.
2.  Identifique o propósito geral dos dados (ex: "Estes dados parecem ser um registro de eventos...", "Esta é uma lista de contatos...").
3.  Agrupe informações relacionadas se possível.
4.  Seja claro e conciso.
5.  O objetivo é que alguém possa ler seu texto e entender o conteúdo da planilha sem precisar vê-la.
6.  **IMPORTANTE:** O seu resultado deve ser APENAS o texto descritivo. Não inclua introduções como "Aqui está o resumo dos dados:" ou qualquer outra frase fora da descrição em si.

**Dados da Planilha (formato CSV):**
{csv_data}

**Descrição Explicativa em Português:**
"""

def translate_excel_advanced(excel_filepath: str, output_txt_filepath: str):
    """
    Lê um arquivo Excel, usa um LLM para gerar uma descrição textual avançada
    e salva o resultado em um arquivo .txt.
    """
    app_logger.info(f"Iniciando TRADUÇÃO AVANÇADA do arquivo Excel: {excel_filepath}")

    if not os.path.exists(excel_filepath):
        app_logger.error(f"Arquivo Excel não encontrado em: {excel_filepath}")
        return

    try:
        xls = pd.ExcelFile(excel_filepath)
        ollama_client = ollama.Client(host=OLLAMA_HOST)
    except Exception as e:
        app_logger.error(f"Erro ao inicializar dependências (pandas ou ollama): {e}", exc_info=True)
        return

    full_text_description = []
    app_logger.info(f"Planilhas encontradas: {xls.sheet_names}")

    for sheet_name in xls.sheet_names:
        app_logger.info(f"Processando planilha: '{sheet_name}'")
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str).fillna("")
        except Exception as e:
            app_logger.error(f"Erro ao ler a planilha '{sheet_name}': {e}", exc_info=True)
            continue
        
        if df.empty:
            app_logger.warning(f"Planilha '{sheet_name}' está vazia. Pulando.")
            continue

        full_text_description.append(f"Resumo da Planilha: '{sheet_name}'\n\n")

        # Processa o DataFrame em chunks de linhas
        num_chunks = (len(df) + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
        
        for i in tqdm(range(num_chunks), desc=f"Analisando Chunks da Planilha '{sheet_name}'"):
            start_row = i * ROWS_PER_CHUNK
            end_row = start_row + ROWS_PER_CHUNK
            df_chunk = df[start_row:end_row]

            # Converte o chunk do DataFrame para uma string CSV
            csv_data_string = df_chunk.to_csv(index=False)
            
            # Formata o prompt
            prompt = PROMPT_TEMPLATE_AVANCADO.format(csv_data=csv_data_string)
            
            app_logger.info(f"Enviando chunk {i+1}/{num_chunks} da planilha '{sheet_name}' para o LLM ({LLM_MODEL})...")
            
            try:
                response = ollama_client.chat(
                    model=LLM_MODEL,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                chunk_description = response['message']['content']
                full_text_description.append(chunk_description)
                full_text_description.append("\n\n---\n\n") # Separador entre descrições de chunks
                app_logger.info(f"Descrição do chunk {i+1} recebida com sucesso.")
            except Exception as e:
                error_message = f"Erro ao comunicar com o LLM para o chunk {i+1}: {e}"
                app_logger.error(error_message, exc_info=True)
                full_text_description.append(f"[[ERRO AO PROCESSAR ESTA PARTE DOS DADOS: {error_message}]]\n\n---\n\n")

    if not full_text_description:
        app_logger.warning(f"Nenhuma descrição gerada para o arquivo Excel: {excel_filepath}")
        return

    try:
        output_dir = os.path.dirname(output_txt_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_txt_filepath, 'w', encoding='utf-8') as f:
            f.write("".join(full_text_description))
        app_logger.info(f"Tradução avançada salva com sucesso em: {output_txt_filepath}")
        print(f"Arquivo de texto gerado salvo em: {output_txt_filepath}")
    except Exception as e:
        app_logger.error(f"Erro ao salvar o arquivo de texto {output_txt_filepath}: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traduz arquivos Excel para texto explicativo usando um LLM."
    )
    parser.add_argument(
        "--input_excel",
        type=str,
        required=True,
        help="Caminho para o arquivo Excel de entrada (.xlsx ou .xls)."
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        required=True,
        help="Caminho para salvar o arquivo de texto (.txt) de saída."
    )
    args = parser.parse_args()
    translate_excel_advanced(args.input_excel, args.output_txt)