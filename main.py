# main.py
import argparse
import sys
import os
import time
import lancedb # Importar para a verificação da DB

# Assegure-se que config.py e outros módulos .py estejam no mesmo diretório
# ou que o Python possa encontrá-los (PYTHONPATH ou estrutura do projeto)
from config import DOCUMENTS_DIR, OLLAMA_HOST, VECTOR_DB_PATH, VECTOR_DB_TABLE_NAME
from processador_documentos import load_documents_from_directory
from rag_pipeline import RAGPipeline
from utils import app_logger # app_logger configurado em utils.py também imprime no terminal

print("DEBUG: Script main.py INICIADO.")

def handle_ingestion(rag_pipe: RAGPipeline):
     print("DEBUG: Função handle_ingestion INICIADA.")
     app_logger.info(f"Iniciando ingestão de documentos do diretório: {DOCUMENTS_DIR}")
    
     if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
         app_logger.error(f"Diretório de documentos '{DOCUMENTS_DIR}' não encontrado ou está vazio.")
         print(f"DEBUG: Erro em handle_ingestion: Diretório de documentos '{DOCUMENTS_DIR}' não encontrado ou está vazio.")
         app_logger.error("Por favor, crie o diretório e adicione seus arquivos .pdf, .docx, .txt, .csv.")
         print("DEBUG: Função handle_ingestion FINALIZADA (erro de diretório).")
         return

     print("DEBUG: handle_ingestion - Carregando documentos...")
     documents = load_documents_from_directory(DOCUMENTS_DIR)
     if not documents:
         app_logger.warning("Nenhum documento foi carregado. Verifique o diretório e os formatos dos arquivos.")
         print("DEBUG: handle_ingestion - Nenhum documento carregado.")
         print("DEBUG: Função handle_ingestion FINALIZADA (sem documentos).")
         return
    
     print(f"DEBUG: handle_ingestion - {len(documents)} documentos carregados. Iniciando ingestão no RAG pipeline...")
     rag_pipe.ingest_documents(documents) # Esta função em rag_pipeline.py também deve ter logs/prints
     app_logger.info("Ingestão de documentos concluída.")
     print("DEBUG: Função handle_ingestion FINALIZADA (sucesso).")


def handle_query_cli(rag_pipe: RAGPipeline):
    print("DEBUG: Função handle_query_cli INICIADA.")
    app_logger.info("Iniciando CLI de Perguntas e Respostas. Digite 'sair' ou 'exit' para terminar.")
    print("\nBem-vindo ao Agente de Base de Conhecimento Corporativo!")
    
    llm_model_name_display = "N/A (verifique Ollama)"
    embedding_model_name_display = "N/A"
    db_uri_display = "N/A"

    if rag_pipe.ollama_client:
        try:
            # O check inicial em main() já deve ter garantido que Ollama está acessível
            # e que LLM_MODEL em config.py está disponível ou foi baixado.
            # RAGPipeline deve armazenar o nome do modelo que está usando.
            if hasattr(rag_pipe, 'LLM_MODEL'): # Se RAGPipeline armazena LLM_MODEL
                 llm_model_name_display = rag_pipe.LLM_MODEL
            else: # Caso contrário, pegue de config (assumindo que rag_pipe usa o de config)
                 from config import LLM_MODEL as config_llm_model
                 llm_model_name_display = config_llm_model
        except Exception as e_ollama_cli:
            print(f"DEBUG: handle_query_cli - Não foi possível obter nome do modelo LLM: {e_ollama_cli}")
            app_logger.warning(f"Não foi possível obter nome do modelo LLM: {e_ollama_cli}")
    
    if rag_pipe.embedding_model: # sentence_transformers model object
        # O nome do modelo de embedding é guardado em config.py e usado para carregar.
        # RAGPipeline deve armazenar o nome que usou.
        if hasattr(rag_pipe, 'EMBEDDING_MODEL_NAME'): # Se RAGPipeline armazena EMBEDDING_MODEL_NAME
            embedding_model_name_display = rag_pipe.EMBEDDING_MODEL_NAME
        else:
            from config import EMBEDDING_MODEL_NAME as config_embedding_model
            embedding_model_name_display = config_embedding_model


    if rag_pipe.db_conn:
        db_uri_display = rag_pipe.db_conn.uri

    print(f"Conectado ao LLM (Ollama): {llm_model_name_display}")
    print(f"Usando modelo de embedding: {embedding_model_name_display}")
    print(f"Consultando base em: {db_uri_display}")
    print("----------------------------------------------------")

    while True:
        try:
            query = input("\nSua pergunta: ")
            if query.lower() in ["sair", "exit", "quit", "q"]:
                app_logger.info("Saindo da CLI.")
                break
            if not query.strip():
                continue

            print(f"DEBUG: handle_query_cli - Query recebida: '{query}'")
            start_time = time.time()
            answer = rag_pipe.answer_query(query) # Esta função em rag_pipeline.py deve ter logs/prints
            end_time = time.time()

            print(f"\nResposta (em {end_time - start_time:.2f}s):")
            print(answer)
            print("----------------------------------------------------")

        except KeyboardInterrupt:
            app_logger.info("Interrupção pelo usuário. Saindo...")
            print("\nDEBUG: handle_query_cli - KeyboardInterrupt recebido.")
            break
        except Exception as e:
            app_logger.error(f"Erro durante o loop de query: {e}", exc_info=True)
            print(f"DEBUG: handle_query_cli - Erro no loop de query: {e}")
            print("Ocorreu um erro. Verifique os logs. Tente novamente ou saia.")
    print("DEBUG: Função handle_query_cli FINALIZADA.")


def main():
    print("DEBUG: Função main() INICIADA.")
    parser = argparse.ArgumentParser(description="Agente de Base de Conhecimento Local Corporativo")
    parser.add_argument(
        "command",
        choices=["ingest", "ask"],
        help="Comando a ser executado: 'ingest' para processar documentos, 'ask' para iniciar a CLI de perguntas."
    )

    args = None
    try:
        print("DEBUG: Antes de parser.parse_args()")
        args = parser.parse_args()
        print(f"DEBUG: Comando recebido: {args.command}")
    except SystemExit as e:
        print(f"DEBUG: Erro no argparse (SystemExit): {e}. Provavelmente argumento inválido ou faltando.")
        if args is None: # Se o parse falhou completamente
             print("DEBUG: args é None após parse_args, saindo de main().")
             return
    
    # Verifica a disponibilidade do Ollama
    try:
        print("DEBUG: Verificando Ollama...")
        import ollama # Importa a biblioteca ollama
        # A instância do cliente Ollama será criada dentro da RAGPipeline
        # Mas podemos fazer um check rápido aqui se quisermos
        client = ollama.Client(host=OLLAMA_HOST)
        client.list() # Verifica se consegue listar modelos, indica que Ollama está respondendo
        app_logger.info(f"Ollama detectado e acessível em {OLLAMA_HOST}.")
        print("DEBUG: Ollama OK.")
    except ImportError:
        app_logger.error("Biblioteca 'ollama' não encontrada. Por favor, instale com 'pip install ollama'.")
        print("DEBUG: Erro fatal - Biblioteca 'ollama' não encontrada.")
        sys.exit(1)
    except Exception as e: # Captura outros erros de conexão com Ollama
        app_logger.error(f"Não foi possível conectar ao Ollama em {OLLAMA_HOST}. Verifique se está em execução.")
        app_logger.error(f"Erro: {e}", exc_info=True)
        print(f"DEBUG: Erro ao conectar com Ollama: {e}")
        print("Certifique-se de que o Ollama está instalado, em execução (`ollama serve`) e acessível.")
        print("Você pode baixá-lo em https://ollama.com/")
        print("DEBUG: Função main() FINALIZADA (erro Ollama).")
        sys.exit(1)

    rag_pipeline_instance = None
    try:
        print("DEBUG: Antes de instanciar RAGPipeline()")
        rag_pipeline_instance = RAGPipeline() # RAGPipeline.__init__ também deve ter prints de DEBUG
        print("DEBUG: RAGPipeline() instanciada com sucesso.")

        if args.command == "ingest":
            print("DEBUG: Comando 'ingest' selecionado.")
            handle_ingestion(rag_pipeline_instance)
        elif args.command == "ask":
            print("DEBUG: Comando 'ask' selecionado.")
            db_exists = False
            print(f"DEBUG: Verificando existência do DB em: {VECTOR_DB_PATH} e tabela '{VECTOR_DB_TABLE_NAME}'")
            
            if rag_pipeline_instance.db_conn:
                print(f"DEBUG: Conexão com DB (rag_pipeline_instance.db_conn) existe: {rag_pipeline_instance.db_conn.uri}")
                try:
                    table_names_in_db = rag_pipeline_instance.db_conn.table_names()
                    print(f"DEBUG: Tabelas no DB: {table_names_in_db}")
                    if VECTOR_DB_TABLE_NAME in table_names_in_db:
                        table = rag_pipeline_instance.db_conn.open_table(VECTOR_DB_TABLE_NAME)
                        table_length = 0
                        try: # Tenta obter o número de linhas
                            table_length = table.to_lance().count_rows() # Forma recomendada e eficiente
                        except Exception as count_err:
                            print(f"DEBUG: Falha ao usar table.to_lance().count_rows(): {count_err}. Tentando len(table)...")
                            try:
                                table_length = len(table) # Pode ser menos eficiente para tabelas grandes
                            except Exception as len_err:
                                print(f"DEBUG: Falha ao usar len(table): {len_err}. Verificando se há pelo menos 1 item.")
                                # Verifica se há pelo menos um item de forma mais leve
                                if next(table.search().limit(1).to_arrow(batch_size=1).to_reader(), None) is not None:
                                    table_length = 1 # Indica que a tabela não está vazia
                                else:
                                    table_length = 0
                        
                        print(f"DEBUG: Tabela '{table.name}' aberta, contagem de linhas (ou indicador de >0): {table_length}")
                        if table_length > 0:
                            db_exists = True
                        else:
                            app_logger.warning(f"A tabela '{table.name}' existe mas está vazia (0 linhas).")
                            print(f"DEBUG: Tabela '{table.name}' existe mas está vazia.")
                    else:
                        app_logger.warning(f"A tabela '{VECTOR_DB_TABLE_NAME}' não foi encontrada no banco de dados ({table_names_in_db}).")
                        print(f"DEBUG: Tabela '{VECTOR_DB_TABLE_NAME}' não encontrada nas tabelas existentes: {table_names_in_db}")
                except lancedb.common.LanceDBClientError as e_lancedb: # Erro específico do LanceDB
                    app_logger.warning(f"Erro LanceDB ao verificar tabela: {e_lancedb}", exc_info=True)
                    print(f"DEBUG: Erro LanceDB ao verificar tabela: {e_lancedb}")
                except Exception as e_tbl: # Outros erros
                    app_logger.error(f"Erro genérico ao verificar tabela: {e_tbl}", exc_info=True)
                    print(f"DEBUG: Erro genérico ao verificar tabela: {e_tbl}")
            else:
                print("DEBUG: rag_pipeline_instance.db_conn é None. A conexão com DB não foi estabelecida na RAGPipeline.")

            print(f"DEBUG: db_exists = {db_exists}")
            if not db_exists:
                 app_logger.warning("A base de conhecimento parece estar vazia ou não foi criada.")
                 print("DEBUG: DB não existe ou está vazia. Mostrando mensagem para o usuário.")
                 print("\nA base de conhecimento está vazia ou não foi criada.")
                 print("Por favor, execute o comando 'ingest' primeiro: python main.py ingest")
            else:
                print("DEBUG: DB existe e tem dados. Chamando handle_query_cli().")
                handle_query_cli(rag_pipeline_instance)

    except RuntimeError as e: # Erros críticos como modelo LLM não encontrado na RAGPipeline
        print(f"DEBUG: RuntimeError capturado em main(): {e}")
        app_logger.critical(f"Erro crítico de runtime: {e}", exc_info=True)
        print(f"Erro crítico: {e}. Verifique os logs e as instruções de configuração.")
    except Exception as e: # Outros erros inesperados
        print(f"DEBUG: Exception genérica capturada em main(): {e}")
        app_logger.critical(f"Ocorreu um erro inesperado no nível principal: {e}", exc_info=True)
        print(f"Um erro inesperado ocorreu: {e}. Consulte o arquivo agent.log para detalhes.")
    finally:
        print("DEBUG: Bloco finally em main() alcançado.")
        if rag_pipeline_instance:
            print("DEBUG: Chamando rag_pipeline_instance.close()")
            rag_pipeline_instance.close() # rag_pipeline.py deve ter o método close()
        # app_logger.info("Aplicação finalizada.") # Loguru já imprime no stderr, não precisa duplicar com print
        print("DEBUG: Aplicação finalizada (do bloco finally de main()).")

if __name__ == "__main__":
    print("DEBUG: Bloco if __name__ == '__main__' ALCANÇADO.")
    main()
    print("DEBUG: Script main.py FINALIZADO (após chamada main()).")