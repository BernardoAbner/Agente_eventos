# Agente de Base de Conhecimento Local Corporativo

Este projeto implementa um agente de Inteligência Artificial local capaz de acessar e processar uma base de conhecimento interna de uma empresa. Ele opera offline em notebooks corporativos com 8GB de RAM (ou mais), utilizando apenas CPU, e é construído com ferramentas gratuitas e de código aberto.

## Funcionalidades

* Processa e indexa documentos nos formatos: PDF (baseados em texto), DOCX (baseados em texto), TXT e CSV.
* Permite que usuários façam perguntas em linguagem natural sobre o conteúdo dos documentos indexados.
* Fornece respostas baseadas exclusivamente nas informações contidas nos documentos.
* Opera totalmente offline após a configuração inicial e download dos modelos.
* Garante a privacidade dos dados, mantendo todos os documentos, índices e interações localmente.
* Interface de Linha de Comando (CLI) interativa.
* (Opcional) Suporte a OCR para PDFs de imagem e outros formatos (requer instalação de Tesseract e Pandoc).

## Componentes Principais

* **Motor LLM Local:** Ollama (gerenciando Phi-3 Mini ou TinyLlama).
* **Modelo de Embedding:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`).
* **Banco de Dados Vetorial:** LanceDB.
* **Processamento de Arquivos:** `pypdf`, `python-docx`, `csv` (Python nativo).
* **(Opcional OCR):** `kreuzberg` (com Tesseract e Pandoc).

## Pré-requisitos

1.  **Python 3.9+:** [Download Python](https://www.python.org/downloads/).
2.  **Ollama:** Siga as instruções de instalação em [ollama.com](https://ollama.com/). Certifique-se de que o serviço Ollama esteja em execução (`ollama serve` em um terminal separado, ou via serviço de sistema).

## Instruções de Instalação e Configuração

1.  **Clone ou Baixe o Projeto:**
    Obtenha os arquivos do projeto e coloque-os em um diretório local (ex: `corporate_kb_agent`).

2.  **Crie e Ative um Ambiente Virtual Python:**
    ```bash
    cd corporate_kb_agent
    python -m venv venv
    ```
    * No Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * No macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Instale as Dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os Modelos LLM via Ollama:**
    * **Phi-3 Mini (Primário):**
        ```bash
        ollama pull phi-3:mini-4k-instruct-q4_K_M
        ```
    * **(Opcional - Contingência LLM) TinyLlama:**
        ```bash
        ollama pull tinyllama:1.1b-chat-q4_K_M
        ```
    O modelo de embedding (`BAAI/bge-small-en-v1.5`) será baixado automaticamente pela biblioteca `sentence-transformers` na primeira execução que o necessitar e ficará em cache para uso offline.

5.  **(Opcional - Configuração para OCR)**
    Se você precisar processar PDFs baseados em imagem ou outros formatos que exijam OCR, e se for permitido em seu ambiente corporativo:
    * **Instale Tesseract-OCR:** Siga as instruções para seu sistema operacional em [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Adicione `tesseract` ao PATH do sistema.
    * **Instale Pandoc:** Siga as instruções em [pandoc.org](https://pandoc.org/installing.html). Adicione `pandoc` ao PATH do sistema.
    * **Modifique `config.py`:** Altere `ENABLE_OCR = False` para `ENABLE_OCR = True`.
    * **Instale a dependência Python para OCR:**
        ```bash
        pip install "kreuzberg[ocr]"
        ```
        (Considere adicionar `"kreuzberg[ocr]"` ao `requirements.txt` se for usar esta funcionalidade permanentemente).

## Uso do Agente

1.  **Prepare seus Documentos:**
    * Crie um subdiretório chamado `knowledge_base_documents` dentro do diretório principal do projeto (`corporate_kb_agent`).
    * Copie os arquivos PDF, DOCX, TXT e CSV que você deseja que o agente processe para este diretório.

2.  **Ingestão de Documentos (Indexação):**
    Execute o script para processar e indexar os documentos. Este processo pode levar algum tempo dependendo do volume e tamanho dos arquivos.
    ```bash
    python main.py ingest
    ```
    Os dados processados (índice vetorial) serão armazenados no subdiretório `data/lancedb`.

3.  **Consultar a Base de Conhecimento:**
    Após a ingestão, inicie a interface de linha de comando para fazer perguntas:
    ```bash
    python main.py ask
    ```
    Digite sua pergunta e pressione Enter. O agente buscará informações nos documentos indexados e gerará uma resposta.
    Para sair, digite `sair`, `exit` ou `quit`.

## Configuração Avançada (Opcional)

Você pode ajustar diversos parâmetros no arquivo `config.py`:

* `LLM_MODEL`: Para trocar entre `phi-3:mini-4k-instruct-q4_K_M` e `tinyllama:1.1b-chat-q4_K_M`.
* `EMBEDDING_MODEL_NAME`: Para trocar o modelo de embedding (ex: para `all-MiniLM-L6-v2`).
* `CHUNK_SIZE`, `CHUNK_OVERLAP`: Para ajustar como os documentos são divididos. (Requer re-ingestão).
* `TOP_K_RESULTS`: Número de chunks de texto mais relevantes a serem recuperados para responder a uma pergunta.
* `ENABLE_OCR`: Para habilitar/desabilitar a funcionalidade de OCR.

## Privacidade de Dados

Todos os documentos, textos extraídos, embeddings e logs de consulta permanecem **exclusivamente na máquina local do usuário**. Nenhuma informação é enviada para serviços externos.

## Gerenciamento de Memória

O sistema foi projetado para operar em máquinas com 8GB de RAM.
* Ollama gerencia o modelo LLM em um processo separado.
* São utilizados modelos quantizados (q4_K_M) e modelos de embedding leves (bge-small).
* LanceDB é otimizado para uso eficiente de RAM com persistência em disco.
* Se encontrar problemas de memória, considere:
    * Usar o LLM `TinyLlama` (menor).
    * Reduzir `CHUNK_SIZE` em `config.py` (e re-ingerir).
    * Fechar outras aplicações que consomem muita RAM.

## Logs

Logs detalhados da operação do agente são salvos em `data/agent.log`.

## Solução de Problemas Comuns

* **Erro de conexão com Ollama:** Certifique-se de que o serviço Ollama está em execução. Execute `ollama serve` em um terminal ou verifique se o serviço de desktop está ativo. Verifique também se `OLLAMA_HOST` em `config.py` (ou a variável de ambiente) está correto.
* **Modelo LLM não encontrado:** Use `ollama list` para ver os modelos baixados. Use `ollama pull <nome_do_modelo>` para baixá-los.
* **Problemas de OCR:** Confirme que Tesseract e Pandoc estão instalados e no PATH do sistema. Verifique os logs para mensagens de erro específicas do `kreuzberg`.
* **"Base de conhecimento vazia" ao usar `ask`:** Certifique-se de que executou `python main.py ingest` primeiro e que havia documentos no diretório `knowledge_base_documents`.