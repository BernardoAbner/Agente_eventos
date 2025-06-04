import pandas as pd
import argparse
import os
from utils import app_logger # Usaremos o mesmo logger do projeto principal

def translate_excel_to_text(excel_filepath: str, output_txt_filepath: str):
    """
    Lê um arquivo Excel, converte seu conteúdo para linguagem natural textual
    e salva em um arquivo .txt.
    """
    app_logger.info(f"Iniciando tradução do arquivo Excel: {excel_filepath}")

    if not os.path.exists(excel_filepath):
        app_logger.error(f"Arquivo Excel não encontrado em: {excel_filepath}")
        print(f"Erro: Arquivo Excel não encontrado em: {excel_filepath}")
        return

    try:
        xls = pd.ExcelFile(excel_filepath)
    except FileNotFoundError:
        app_logger.error(f"Arquivo Excel não encontrado (verificado novamente pelo pandas): {excel_filepath}")
        print(f"Erro: Arquivo Excel não encontrado: {excel_filepath}")
        return
    except Exception as e:
        app_logger.error(f"Erro ao abrir o arquivo Excel {excel_filepath}: {e}")
        print(f"Erro ao abrir o arquivo Excel {excel_filepath}: {e}")
        return

    full_text_description = []
    app_logger.info(f"Planilhas encontradas: {xls.sheet_names}")

    for sheet_name in xls.sheet_names:
        app_logger.info(f"Processando planilha: '{sheet_name}'")
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str) # Ler tudo como string
            df.fillna("", inplace=True) # Substituir NaN por string vazia para consistência
        except Exception as e:
            app_logger.error(f"Erro ao ler a planilha '{sheet_name}' do arquivo {excel_filepath}: {e}")
            continue # Pula para a próxima planilha

        if df.empty:
            app_logger.warning(f"Planilha '{sheet_name}' está vazia. Pulando.")
            full_text_description.append(f"Informação da Planilha: '{sheet_name}' (Vazia)\n")
            continue

        full_text_description.append(f"Conteúdo da Planilha: '{sheet_name}'\n")

        # Adiciona os nomes das colunas como uma "linha de cabeçalho" textual
        header_text = "As colunas são: " + ", ".join([str(col) for col in df.columns]) + "."
        full_text_description.append(header_text)

        for index, row in df.iterrows():
            row_description_parts = []
            # Usar um prefixo para cada linha pode ajudar o LLM a entender a estrutura
            prefix = f"No registro {index + 1} da planilha '{sheet_name}': "

            for col_name in df.columns:
                cell_value = str(row[col_name]).strip()
                if cell_value: # Apenas adiciona se a célula não estiver vazia
                    row_description_parts.append(f"{str(col_name)} é '{cell_value}'")

            if row_description_parts:
                full_text_description.append(prefix + ", ".join(row_description_parts) + ".")
            else:
                # Se a linha inteira estiver vazia (após preencher NaNs e strip), pode ser útil registrar.
                full_text_description.append(f"No registro {index + 1} da planilha '{sheet_name}': (linha aparentemente vazia ou sem dados significativos).")
        
        full_text_description.append("\n--- Fim da Planilha '" + sheet_name + "' ---\n")
        app_logger.info(f"Planilha '{sheet_name}' processada. {len(df.index)} linhas lidas.")

    if not full_text_description:
        app_logger.warning(f"Nenhum conteúdo extraído do arquivo Excel: {excel_filepath}")
        print(f"Aviso: Nenhum conteúdo foi extraído de {excel_filepath}")
        return

    try:
        # Garante que o diretório de saída exista
        output_dir = os.path.dirname(output_txt_filepath)
        if output_dir and not os.path.exists(output_dir):
            app_logger.info(f"Criando diretório de saída: {output_dir}")
            os.makedirs(output_dir)
            
        with open(output_txt_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_text_description))
        app_logger.info(f"Tradução do Excel salva com sucesso em: {output_txt_filepath}")
        print(f"Arquivo de texto gerado salvo em: {output_txt_filepath}")
    except Exception as e:
        app_logger.error(f"Erro ao salvar o arquivo de texto {output_txt_filepath}: {e}")
        print(f"Erro ao salvar o arquivo de texto: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Traduz arquivos Excel para formato de texto em linguagem natural para ingestão pelo Agente RAG."
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
    # Você pode adicionar um argumento para o diretório de documentos se quiser padronizar
    # parser.add_argument(
    #     "--docs_dir",
    #     type=str,
    #     default="knowledge_base_documents",
    #     help="Diretório onde o arquivo .txt traduzido será salvo (nome do arquivo será derivado do input)."
    # )

    args = parser.parse_args()

    # Validação simples dos caminhos
    # if not args.output_txt.endswith(".txt"):
    #     print("Erro: O arquivo de saída deve ter a extensão .txt")
    #     app_logger.error("Nome de arquivo de saída inválido, não termina com .txt")
    #     return

    translate_excel_to_text(args.input_excel, args.output_txt)

if __name__ == "__main__":
    # Nota: O logger configurado em utils.py será usado.
    # Se utils.py não for encontrado (ex: rodando este script de um diretório diferente),
    # você pode adicionar um logger básico aqui como fallback.
    main()