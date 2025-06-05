# Licenças de Software das Dependências do Projeto

Este projeto utiliza as seguintes bibliotecas de código aberto. Agradecemos aos desenvolvedores e mantenedores por seu trabalho.

## Dependências Diretas Principais

* **Python:**
    * Licença: Python Software Foundation License (PSF)
    * URL: [https://docs.python.org/3/license.html](https://docs.python.org/3/license.html)
    * Nota: Permissiva, similar à GPL mas compatível com projetos de código fechado.

* **Ollama (Software e Modelos Gerenciados):**
    * Software Ollama: MIT License
        * URL: [https://github.com/ollama/ollama/blob/main/LICENSE](https://github.com/ollama/ollama/blob/main/LICENSE)
    * Modelo Microsoft Phi-3 Mini: MIT License
        * URL: (Normalmente especificada na página do modelo no Hugging Face ou site da Microsoft) - Confirme a licença específica da versão utilizada. Geralmente permissiva para os modelos Phi.
    * Modelo TinyLlama: Apache 2.0 License
        * URL: [https://github.com/jzhang38/TinyLlama/blob/main/LICENSE](https://github.com/jzhang38/TinyLlama/blob/main/LICENSE)

* **sentence-transformers (Biblioteca):**
    * Licença: Apache 2.0 License
    * URL: [https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
    * Modelo `BAAI/bge-small-en-v1.5`: MIT License (Verifique na página do modelo no Hugging Face)
        * URL: [https://huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
    * Modelo `all-MiniLM-L6-v2`: Apache 2.0 License (Verifique na página do modelo no Hugging Face)
        * URL: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

* **LanceDB (Biblioteca):**
    * Licença: Apache 2.0 License
    * URL: [https://github.com/lancedb/lancedb/blob/main/LICENSE](https://github.com/lancedb/lancedb/blob/main/LICENSE)

* **pypdf (Biblioteca):**
    * Licença: BSD 3-Clause "New" or "Revised" License
    * URL: [https://github.com/py-pdf/pypdf/blob/main/LICENSE](https://github.com/py-pdf/pypdf/blob/main/LICENSE)

* **python-docx (Biblioteca):**
    * Licença: MIT License
    * URL: [https://github.com/python-openxml/python-docx/blob/master/LICENSE](https://github.com/python-openxml/python-docx/blob/master/LICENSE)

* **loguru (Biblioteca):**
    * Licença: MIT License
    * URL: [https://github.com/Delgan/loguru/blob/master/LICENSE](https://github.com/Delgan/loguru/blob/master/LICENSE)

* **tqdm (Biblioteca):**
    * Licença: MIT License / MPL 2.0 (Dual-licensed)
    * URL: [https://github.com/tqdm/tqdm/blob/master/LICENCE](https://github.com/tqdm/tqdm/blob/master/LICENCE)

* **python-dotenv (Biblioteca):**
    * Licença: BSD 3-Clause "New" or "Revised" License
    * URL: [https://github.com/theskumar/python-dotenv/blob/main/LICENSE](https://github.com/theskumar/python-dotenv/blob/main/LICENSE)

* **ollama (Python client library):**
    * Licença: MIT License
    * URL: [https://github.com/ollama/ollama-python/blob/main/LICENSE](https://github.com/ollama/ollama-python/blob/main/LICENSE)


* **(Opcional) kreuzberg (Biblioteca para OCR):**
    * Licença: Apache 2.0 License
    * URL: [https://github.com/Datatera/kreuzberg/blob/main/LICENSE](https://github.com/Datatera/kreuzberg/blob/main/LICENSE)
    * Nota: `kreuzberg` pode usar backends como Tesseract-OCR, EasyOCR, PaddleOCR, cada um com suas próprias licenças e dependências.
        * Tesseract-OCR: Apache 2.0 License
        * Pandoc (se usado por `kreuzberg` indiretamente): GPLv2 or later

## Dependências Transitivas Comuns

As bibliotecas acima podem ter suas próprias dependências. Algumas comuns e suas licenças típicas:

* **NumPy:** BSD 3-Clause License
* **SciPy:** BSD 3-Clause License
* **Pandas:** BSD 3-Clause License
* **PyArrow (usado por LanceDB):** Apache 2.0 License
* **Requests:** Apache 2.0 License
* **urllib3:** MIT License
* **Hugging Face Hub (usado por `sentence-transformers`):** Apache 2.0 License
* **transformers (usado por `sentence-transformers`):** Apache 2.0 License
* **tokenizers (usado por `sentence-transformers`):** Apache 2.0 License

## Importante

* Esta lista é um resumo e pode não ser exaustiva para todas as dependências transitivas.
* É responsabilidade do usuário verificar as licenças de todas as dependências (diretas e transitivas) para garantir a conformidade com os requisitos corporativos. Ferramentas como `pip-licenses` (`pip install pip-licenses && pip-licenses --from=mixed --format=markdown`) podem ajudar a gerar uma lista mais completa.
* As licenças dos modelos de linguagem e embedding (especialmente aqueles baixados do Hugging Face) devem ser verificadas em suas respectivas páginas de modelo, pois podem variar. As licenças MIT e Apache 2.0 são geralmente permissivas para uso corporativo.

Este arquivo `LICENSES.md` deve ser revisado e atualizado conforme o projeto evolui ou as dependências mudam.