# rag_pipeline.py
import ollama
import lancedb
from sentence_transformers import SentenceTransformer # Certifique-se que este import está aqui
from tqdm import tqdm
import gc

# Imports de config devem estar acessíveis
from config import (
    LLM_MODEL, EMBEDDING_MODEL_NAME, VECTOR_DB_PATH, VECTOR_DB_TABLE_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, PROMPT_TEMPLATE, OLLAMA_HOST
)
from utils import app_logger

print("DEBUG: Script rag_pipeline.py INICIADO.")

class RAGPipeline:  # Nível 0 de indentação
    # Todos os métodos da classe (def) devem começar com 4 espaços de indentação aqui
    def __init__(self): # 4 espaços de indentação
        print("DEBUG: RAGPipeline __init__ INICIADO.")
        app_logger.info("Inicializando RAGPipeline...")
        
        # Atributos da instância
        self.embedding_model = None
        self.db_conn = None
        self.table = None
        
        # Atributos baseados na configuração para fácil acesso dentro da classe
        self.LLM_MODEL = LLM_MODEL
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.VECTOR_DB_PATH = VECTOR_DB_PATH # Embora usado diretamente de config às vezes, bom ter aqui
        self.VECTOR_DB_TABLE_NAME = VECTOR_DB_TABLE_NAME
        self.OLLAMA_HOST = OLLAMA_HOST
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP
        self.TOP_K_RESULTS = TOP_K_RESULTS
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE

        print("DEBUG: RAGPipeline __init__ - Carregando modelo de embedding...")
        self._load_embedding_model()
        print("DEBUG: RAGPipeline __init__ - Conectando ao Vector DB...")
        self._connect_vector_db()
        print("DEBUG: RAGPipeline __init__ - Inicializando cliente Ollama...")
        self.ollama_client = ollama.Client(host=self.OLLAMA_HOST)
        print("DEBUG: RAGPipeline __init__ - Verificando modelo Ollama...")
        self._check_ollama_model()
        print("DEBUG: RAGPipeline __init__ FINALIZADO.")

    # Este método agora está CORRETAMENTE INDENTADO como parte da classe RAGPipeline
# Dentro da classe RAGPipeline em rag_pipeline.py

    def _check_ollama_model(self):
        print("DEBUG: RAGPipeline - _check_ollama_model INICIADO.")
        
 # Dentro da classe RAGPipeline, na função _check_ollama_model:
# Substitua a função auxiliar get_model_names_from_response por esta:

        def get_model_names_from_response(response_data): # response_data é o models_info_response
            names = []
            list_of_model_objects = [] # Lista que VAMOS preencher

            print(f"DEBUG: RAGPipeline (helper) - Tipo de response_data recebido: {type(response_data)}")

            # Cenário 1: response_data é um dicionário com a chave 'models'
            if isinstance(response_data, dict) and 'models' in response_data and isinstance(response_data['models'], list):
                list_of_model_objects = response_data['models']
                print("DEBUG: RAGPipeline (helper) - response_data é um dict, usando response_data['models'].")
            # Cenário 2: response_data é um objeto que tem um atributo 'models' que é uma lista
            # (como ollama._types.ListResponse)
            elif hasattr(response_data, 'models') and isinstance(response_data.models, list):
                list_of_model_objects = response_data.models
                print("DEBUG: RAGPipeline (helper) - response_data é um objeto com atributo .models (lista).")
            # Cenário 3: response_data JÁ É a lista de objetos modelo (menos provável, mas para cobrir)
            elif isinstance(response_data, list):
                list_of_model_objects = response_data
                print("DEBUG: RAGPipeline (helper) - response_data já é uma lista de modelos.")
            else:
                app_logger.warning(f"Formato de response_data ({type(response_data)}) inesperado e não processável: {response_data}")
                print(f"DEBUG: RAGPipeline (helper) - Formato de response_data não reconhecido.")
                return names # Retorna lista vazia

            # Agora itera sobre a list_of_model_objects que foi determinada
            for model_obj in list_of_model_objects:
                model_name_found = None
                # Prioriza o atributo .model se existir, baseado no seu log "Model(model='phi3:mini',...)"
                if hasattr(model_obj, 'model') and isinstance(model_obj.model, str):
                    model_name_found = model_obj.model
                elif hasattr(model_obj, 'name') and isinstance(model_obj.name, str): 
                    model_name_found = model_obj.name
                # Fallback para dicionário (se model_obj for um dict dentro da lista)
                elif isinstance(model_obj, dict):
                    if 'model' in model_obj and isinstance(model_obj['model'], str) : 
                        model_name_found = model_obj['model']
                    elif 'name' in model_obj and isinstance(model_obj['name'], str): 
                        model_name_found = model_obj['name']
                
                if model_name_found:
                    names.append(model_name_found)
                    print(f"DEBUG: RAGPipeline (helper) - Modelo da lista Ollama adicionado: {model_name_found}")
                else:
                    app_logger.warning(f"Entrada de modelo ('{type(model_obj)}') em ollama.list() com formato inesperado ou sem atributo/chave de nome: {model_obj}")
                    print(f"DEBUG: RAGPipeline (helper) - Entrada de objeto de modelo inválida: {model_obj}")
            
            print(f"DEBUG: RAGPipeline (helper) - Nomes de modelos extraídos: {names}")
            return names

        models_info_response_initial = None # Para log em caso de erro
        try:
            if not self.ollama_client:
                app_logger.error("Cliente Ollama não inicializado antes de _check_ollama_model.")
                raise RuntimeError("Ollama client not initialized in RAGPipeline __init__")

            print(f"DEBUG: RAGPipeline - Chamando self.ollama_client.list() (verificação inicial)")
            models_info_response_initial = self.ollama_client.list()
            print(f"DEBUG: RAGPipeline - Resposta inicial de ollama.list(): {models_info_response_initial}")
            
            available_models = get_model_names_from_response(models_info_response_initial)
            
            target_llm = self.LLM_MODEL 
            print(f"DEBUG: RAGPipeline - Verificando por LLM: '{target_llm}' em {available_models}")

            if target_llm not in available_models:
                app_logger.warning(f"Modelo LLM '{target_llm}' não encontrado localmente via Ollama ({available_models}).")
                print(f"DEBUG: RAGPipeline - Modelo '{target_llm}' NÃO encontrado. Tentando pull...")
                
                try:
                    import ollama # Garante que o módulo ollama está disponível para a função pull
                    app_logger.info(f"Tentando baixar/puxar o modelo: ollama pull {target_llm}")
                    ollama.pull(target_llm) # Esta chamada pode demorar
                    app_logger.info(f"Pull do modelo '{target_llm}' solicitado/concluído.")
                    
                    # Re-verificar a lista de modelos APÓS o pull
                    models_info_after_pull = self.ollama_client.list()
                    print(f"DEBUG: RAGPipeline - Resposta de ollama.list() APÓS PULL: {models_info_after_pull}")
                    available_models = get_model_names_from_response(models_info_after_pull) # Usa a função auxiliar
                    print(f"DEBUG: RAGPipeline - Lista de modelos APÓS PULL: {available_models}")

                    if target_llm not in available_models:
                        app_logger.error(f"Modelo LLM '{target_llm}' AINDA não encontrado mesmo após tentativa de pull.")
                        raise RuntimeError(f"Modelo LLM '{target_llm}' não disponível e falha ao efetivar o pull.")
                    else: # Este 'else' corresponde ao 'if target_llm not in available_models' (APÓS o pull)
                        app_logger.info(f"Modelo LLM '{target_llm}' agora disponível após pull.")
                        print(f"DEBUG: RAGPipeline - Modelo '{target_llm}' ENCONTRADO após pull.")
                
                except Exception as e_pull:
                    app_logger.error(f"Falha na operação de pull para o modelo '{target_llm}': {e_pull}", exc_info=True)
                    # Adiciona mais contexto ao erro levantado
                    raise RuntimeError(f"Falha ao tentar baixar o modelo '{target_llm}'. Detalhes: {e_pull}") from e_pull
            
            else: # Este 'else' corresponde ao 'if target_llm not in available_models;' (verificação INICIAL)
                app_logger.info(f"Modelo LLM '{target_llm}' encontrado localmente via Ollama.")
                print(f"DEBUG: RAGPipeline - Modelo '{target_llm}' ENCONTRADO na verificação inicial.")
        
        except Exception as e: # Handler de exceção geral para _check_ollama_model
            app_logger.error(f"Erro geral em _check_ollama_model (Host Ollama: {self.OLLAMA_HOST}): {e!r}", exc_info=True)
            if models_info_response_initial is not None: # Loga a resposta inicial se foi obtida
                 app_logger.error(f"Resposta inicial de ollama.list() (se obtida antes do erro): {models_info_response_initial}")
            print(f"DEBUG: RAGPipeline - Exceção em _check_ollama_model: {e!r}")
            raise # Re-levanta a exceção para ser pega por main.py
        
        print("DEBUG: RAGPipeline - _check_ollama_model FINALIZADO.")

    def _load_embedding_model(self): # 4 espaços de indentação
        print("DEBUG: RAGPipeline _load_embedding_model INICIADO.")
        app_logger.info(f"Carregando modelo de embedding: {self.EMBEDDING_MODEL_NAME}")
        try:
            self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME, trust_remote_code=True)
            app_logger.info("Modelo de embedding carregado com sucesso.")
            print(f"DEBUG: RAGPipeline _load_embedding_model: Modelo '{self.EMBEDDING_MODEL_NAME}' carregado.")
        except Exception as e:
            app_logger.error(f"Falha ao carregar modelo de embedding '{self.EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline _load_embedding_model: ERRO ao carregar '{self.EMBEDDING_MODEL_NAME}': {e}")
            raise
        print("DEBUG: RAGPipeline _load_embedding_model FINALIZADO.")

    def _connect_vector_db(self): # 4 espaços de indentação
        print("DEBUG: RAGPipeline _connect_vector_db INICIADO.")
        app_logger.info(f"Conectando ao banco de dados vetorial em: {self.VECTOR_DB_PATH}")
        try:
            self.db_conn = lancedb.connect(self.VECTOR_DB_PATH)
            app_logger.info("Conexão com LanceDB estabelecida.")
            print(f"DEBUG: RAGPipeline _connect_vector_db: Conexão LanceDB OK para {self.VECTOR_DB_PATH}")
        except Exception as e:
            app_logger.error(f"Falha ao conectar/criar LanceDB em '{self.VECTOR_DB_PATH}': {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline _connect_vector_db: ERRO {e}")
            raise
        print("DEBUG: RAGPipeline _connect_vector_db FINALIZADO.")

    def _simple_text_splitter(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]: # 4 espaços
        print(f"DEBUG: RAGPipeline _simple_text_splitter: size={chunk_size}, overlap={chunk_overlap}, len_text={len(text)}")
        if not text:
            return []
        
        final_chunks = []
        start_idx = 0
        text_length = len(text)
        
        while start_idx < text_length:
            end_idx = min(start_idx + chunk_size, text_length)
            chunk = text[start_idx:end_idx]
            final_chunks.append(chunk)
            
            next_start = start_idx + chunk_size - chunk_overlap
            
            # Se o próximo início for antes ou igual ao atual (overlap >= chunk_size),
            # ou se já estamos no final do texto, quebramos para evitar loop/chunks vazios.
            if next_start <= start_idx or next_start >= text_length:
                break 
            start_idx = next_start
            
        # Caso o último chunk não tenha sido pego completamente devido ao passo de overlap
        # Esta lógica pode ser redundante com o break acima, mas é uma segurança.
        # Se o último end_idx não foi text_length, e ainda há texto
        if final_chunks:
            last_chunk_added_end_idx = text.find(final_chunks[-1]) + len(final_chunks[-1])
            if last_chunk_added_end_idx < text_length:
                remaining_text = text[last_chunk_added_end_idx:]
                if remaining_text.strip(): # Adiciona somente se houver conteúdo útil
                    final_chunks.append(remaining_text)
                    print(f"DEBUG: RAGPipeline _simple_text_splitter: Adicionado chunk final restante de {len(remaining_text)} caracteres.")

        return [c for c in final_chunks if c.strip()]

    def ingest_documents(self, documents: list[dict]): # 4 espaços
        print(f"DEBUG: RAGPipeline ingest_documents: Recebidos {len(documents)} documentos.")
        app_logger.info(f"Iniciando processo de ingestão de {len(documents)} documentos...")
        all_chunks_data = []

        for doc_idx, doc in enumerate(tqdm(documents, desc="Processando Documentos para Ingestão")):
            source_filename = doc['source']
            text_content = doc['content']
            print(f"DEBUG: RAGPipeline ingest_documents: Processando doc {doc_idx+1} '{source_filename}' ({len(text_content)} chars)")

            if not text_content or not text_content.strip():
                app_logger.warning(f"Documento {source_filename} está vazio ou não contém texto. Pulando.")
                continue
            
            app_logger.debug(f"Chunking documento: {source_filename}")
            # Usa o método _simple_text_splitter da classe
            text_chunks = self._simple_text_splitter(text_content, self.CHUNK_SIZE, self.CHUNK_OVERLAP)
            
            app_logger.info(f"Documento '{source_filename}' dividido em {len(text_chunks)} chunks.")
            print(f"DEBUG: RAGPipeline ingest_documents: Doc '{source_filename}' -> {len(text_chunks)} chunks.")

            if not text_chunks:
                app_logger.warning(f"Nenhum chunk gerado para {source_filename}. Pulando.")
                continue

            app_logger.debug(f"Gerando embeddings para chunks de {source_filename}...")
            try:
                # Otimização de memória: processar embeddings em sub-batches se necessário,
                # mas SentenceTransformer lida bem com listas.
                chunk_embeddings = self.embedding_model.encode(
                    text_chunks,
                    show_progress_bar=False, # tqdm já está sendo usado externamente
                    batch_size=32 
                )
                app_logger.debug(f"{len(chunk_embeddings)} embeddings gerados para {source_filename}.")

                for i, chunk_text in enumerate(text_chunks):
                    all_chunks_data.append({
                        "vector": chunk_embeddings[i].tolist(),
                        "text": chunk_text,
                        "source": source_filename,
                        "chunk_num": i + 1
                    })
            except Exception as e:
                app_logger.error(f"Erro ao gerar embeddings para {source_filename}: {e}", exc_info=True)
                print(f"DEBUG: RAGPipeline ingest_documents: Erro embeddings doc '{source_filename}': {e}")
                continue 

            gc.collect()

        if not all_chunks_data:
            app_logger.warning("Nenhum dado para indexar após processar todos os documentos.")
            print("DEBUG: RAGPipeline ingest_documents: Nenhum chunk de dados para indexar.")
            return

        app_logger.info(f"Total de {len(all_chunks_data)} chunks para serem adicionados ao LanceDB.")
        print(f"DEBUG: RAGPipeline ingest_documents: Total {len(all_chunks_data)} chunks para LanceDB.")

        try:
            table_names = self.db_conn.table_names()
            if self.VECTOR_DB_TABLE_NAME in table_names:
                app_logger.info(f"Tabela '{self.VECTOR_DB_TABLE_NAME}' existente. Removendo antes de recriar.")
                print(f"DEBUG: RAGPipeline ingest_documents: Removendo tabela antiga '{self.VECTOR_DB_TABLE_NAME}'.")
                self.db_conn.drop_table(self.VECTOR_DB_TABLE_NAME)
            
            app_logger.info(f"Criando/Recriando tabela '{self.VECTOR_DB_TABLE_NAME}' no LanceDB...")
            print(f"DEBUG: RAGPipeline ingest_documents: Criando tabela '{self.VECTOR_DB_TABLE_NAME}'.")
            # Cria com um item para inferir schema, depois adiciona o resto
            self.table = self.db_conn.create_table(self.VECTOR_DB_TABLE_NAME, data=all_chunks_data[:1], mode="overwrite")
            if len(all_chunks_data) > 1:
                 self.table.add(all_chunks_data[1:])
            app_logger.info(f"Dados ({len(all_chunks_data)} chunks) adicionados à tabela '{self.VECTOR_DB_TABLE_NAME}'.")
            print(f"DEBUG: RAGPipeline ingest_documents: Dados adicionados.")
            
            # Criar índice para otimizar buscas
            # Ajuste os parâmetros conforme o tamanho da sua base de dados e dimensão do embedding
            # if len(all_chunks_data) > 100: # Heurística
            #     app_logger.info("Criando índice IVF_PQ na tabela (pode levar tempo)...")
            #     print("DEBUG: RAGPipeline ingest_documents: Criando índice IVF_PQ...")
            #     try:
            #         # Obter dimensão do embedding dinamicamente
            #         embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            #         num_partitions = min(max(1, int(len(all_chunks_data)**0.5 // 4)), 256) # Ajuste conforme necessidade
            #         num_sub_vectors = embedding_dim // 4 # Comum para IVF_PQ, ajuste se necessário
            #         if num_sub_vectors == 0 : num_sub_vectors = 1 # Evitar divisão por zero ou subvetor zero
            #         if embedding_dim % num_sub_vectors != 0 : # Ajuste para ser divisível
            #             # Encontrar o divisor mais próximo para num_sub_vectors
            #             if embedding_dim > 16 : num_sub_vectors = 16 # Valor comum
            #             else: num_sub_vectors = embedding_dim # Sem subvetores se dimensão muito pequena

            #         if embedding_dim % num_sub_vectors != 0:
            #            app_logger.warning(f"Dimensão do embedding {embedding_dim} não é divisível por num_sub_vectors {num_sub_vectors}. Pulando criação do índice PQ.")
            #         else:
            #             self.table.create_index(metric="L2",
            #                                     num_partitions=num_partitions,
            #                                     num_sub_vectors=num_sub_vectors,
            #                                     replace=True)
            #             app_logger.info(f"Índice IVF_PQ criado (partitions={num_partitions}, sub_vectors={num_sub_vectors}).")
            #             print(f"DEBUG: RAGPipeline ingest_documents: Índice IVF_PQ criado.")
            #     except Exception as e_index:
            #         app_logger.error(f"Falha ao criar índice IVF_PQ: {e_index}. A busca pode ser mais lenta.", exc_info=True)
            #         print(f"DEBUG: RAGPipeline ingest_documents: Erro ao criar índice IVF_PQ: {e_index}")
            # else:
            #     app_logger.info("Número de chunks pequeno, pulando criação de índice IVF_PQ complexo.")
            # Simplificando a criação do índice por agora, LanceDB pode escolher bons defaults.
            if len(all_chunks_data) > 0 : # Só cria índice se tiver dados
                 app_logger.info("Criando índice na tabela (pode levar tempo)...")
                 print("DEBUG: RAGPipeline ingest_documents: Criando índice...")
                 try:
                     self.table.create_index() # Usa defaults do LanceDB, geralmente bom.
                     app_logger.info(f"Índice criado com sucesso.")
                     print(f"DEBUG: RAGPipeline ingest_documents: Índice criado.")
                 except Exception as e_index:
                     app_logger.error(f"Falha ao criar índice: {e_index}. A busca pode ser mais lenta.", exc_info=True)
                     print(f"DEBUG: RAGPipeline ingest_documents: Erro ao criar índice: {e_index}")


        except Exception as e:
            app_logger.error(f"Erro durante a ingestão no LanceDB: {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline ingest_documents: Erro na ingestão no LanceDB: {e}")
            raise
        
        app_logger.info("Processo de ingestão de documentos concluído.")
        print("DEBUG: RAGPipeline ingest_documents FINALIZADO.")

    def retrieve_relevant_chunks(self, query: str) -> list[dict]: # 4 espaços
        print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Query '{query[:30]}...'")
        if not self.table: # Se a tabela não foi setada durante ingest ou __init__ (se 'ask' for rodado primeiro)
            print("DEBUG: RAGPipeline retrieve_relevant_chunks: self.table é None. Tentando abrir.")
            try:
                if self.VECTOR_DB_TABLE_NAME in self.db_conn.table_names():
                    self.table = self.db_conn.open_table(self.VECTOR_DB_TABLE_NAME)
                    app_logger.info(f"Tabela '{self.VECTOR_DB_TABLE_NAME}' aberta com sucesso para retrieve.")
                    print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Tabela '{self.VECTOR_DB_TABLE_NAME}' aberta.")
                else:
                    app_logger.error(f"Tabela '{self.VECTOR_DB_TABLE_NAME}' não encontrada. Execute a ingestão primeiro.")
                    print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Tabela '{self.VECTOR_DB_TABLE_NAME}' não existe.")
                    return []
            except Exception as e_open:
                app_logger.error(f"Erro ao tentar abrir a tabela '{self.VECTOR_DB_TABLE_NAME}': {e_open}", exc_info=True)
                print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Erro ao abrir tabela: {e_open}")
                return []

        app_logger.debug(f"Gerando embedding para a query: '{query[:50]}...'")
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
        except Exception as e:
            app_logger.error(f"Erro ao gerar embedding para a query: {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Erro ao gerar embedding da query: {e}")
            return []

        app_logger.debug(f"Buscando {self.TOP_K_RESULTS} chunks relevantes no LanceDB.")
        try:
            results = self.table.search(query_embedding).limit(self.TOP_K_RESULTS).to_list()
            app_logger.info(f"Encontrados {len(results)} chunks relevantes.")
            print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: {len(results)} chunks encontrados.")
            return results
        except Exception as e:
            app_logger.error(f"Erro ao buscar no LanceDB: {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline retrieve_relevant_chunks: Erro ao buscar no LanceDB: {e}")
            return []

    def generate_response(self, query: str, context_chunks: list[dict]) -> str: # 4 espaços
        print(f"DEBUG: RAGPipeline generate_response: Query '{query[:30]}...', {len(context_chunks)} chunks de contexto.")
        if not context_chunks:
            app_logger.warning("Nenhum chunk de contexto fornecido para generate_response.")
            # O template de prompt já instrui o LLM sobre o que fazer se não houver contexto,
            # mas podemos retornar uma mensagem direta se preferirmos.
            # return "A informação não foi encontrada na base de conhecimento fornecida (nenhum chunk relevante encontrado)."
            # Vamos deixar o LLM decidir com base no prompt.
            pass


        context_str = "\n\n---\n\n".join([
            f"Fonte: {chunk.get('source', 'Desconhecida')}, Chunk {chunk.get('chunk_num', 'N/A')}\n{chunk.get('text', '')}" 
            for chunk in context_chunks
        ])
        
        formatted_prompt = self.PROMPT_TEMPLATE.format(
            contexto_dos_chunks_recuperados=context_str,
            pergunta_do_usuario=query
        )
        app_logger.debug(f"Prompt formatado para LLM (primeiros 300 chars):\n{formatted_prompt[:300]}...")
        print(f"DEBUG: RAGPipeline generate_response: Enviando prompt ao LLM '{self.LLM_MODEL}'.")

        try:
            response = self.ollama_client.chat(
                model=self.LLM_MODEL,
                messages=[{'role': 'user', 'content': formatted_prompt}]
            )
            answer = response['message']['content']
            app_logger.info("Resposta recebida do LLM.")
            app_logger.debug(f"Resposta do LLM: {answer}")
            print("DEBUG: RAGPipeline generate_response: Resposta LLM recebida.")
            return answer
        except Exception as e:
            app_logger.error(f"Erro ao comunicar com o LLM via Ollama: {e}", exc_info=True)
            print(f"DEBUG: RAGPipeline generate_response: Erro ao comunicar com LLM: {e}")
            return "Desculpe, ocorreu um erro ao tentar gerar a resposta (LLM)."

    def answer_query(self, query: str) -> str: # 4 espaços
        print(f"DEBUG: RAGPipeline answer_query: Processando query '{query[:30]}...'")
        app_logger.info(f"Processando query: '{query}'")
        relevant_chunks = self.retrieve_relevant_chunks(query)
        if not relevant_chunks:
            app_logger.warning("Nenhum chunk relevante encontrado para a query.")
            # A generate_response e o PROMPT_TEMPLATE já lidam com contexto vazio.
        
        response = self.generate_response(query, relevant_chunks)
        print("DEBUG: RAGPipeline answer_query FINALIZADO.")
        return response

    def close(self): # 4 espaços
        print("DEBUG: RAGPipeline close INICIADO.")
        app_logger.info("Fechando RAGPipeline...")
        # LanceDB não requer um self.db_conn.close() explícito.
        # A conexão é gerenciada e os dados são persistidos automaticamente.
        app_logger.debug("Conexão LanceDB não requer fechamento explícito.")
        
        if hasattr(self, 'embedding_model') and self.embedding_model:
            # Não há um método close() ou unload() explícito para SentenceTransformer.
            # Deletar a referência e chamar gc.collect() pode ajudar a liberar memória GPU/CPU.
            del self.embedding_model
            self.embedding_model = None
            app_logger.info("Referência ao modelo de embedding removida.")
            print("DEBUG: RAGPipeline close: Referência ao embedding_model removida.")
        
        gc.collect()
        app_logger.info("RAGPipeline finalizada.")
        print("DEBUG: RAGPipeline close FINALIZADO.")

print("DEBUG: Script rag_pipeline.py FINALIZADO (após tudo).")