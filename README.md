# 📚 Modelo RAG Local para Análise de PDFs (Local RAG for PDF Analysis)

Este é um projeto pessoal de um sistema RAG (Retrieval-Augmented Generation) desenvolvido para analisar e extrair informações de documentos em formato PDF. O grande diferencial deste projeto é que ele roda **100% localmente**, garantindo privacidade e autonomia, utilizando modelos via Ollama.

## ⚙️ Pré-requisitos: Configurando a IA Local

Como este projeto não utiliza APIs externas na nuvem, precisamos instalar e rodar os modelos localmente usando o **Ollama**. 

1. Baixe e instale o Ollama em: [https://ollama.com/download](https://ollama.com/download)
2. Após a instalação, abra o seu terminal e execute os três comandos abaixo para baixar os modelos necessários e iniciar o servidor:

# Baixa o modelo de linguagem (LLM) que vai gerar as respostas
	```bash
	ollama pull mistral

# Baixa o modelo de embeddings responsável por vetorizar os textos
	```bash
	ollama pull nomic-embed-text

# Inicia o servidor do Ollama (caso não inicie automaticamente)
	```bash
	ollama serve


## 📂 Estrutura do Projeto

Os arquivos deste repositório estão organizados da seguinte forma:

* **`get_embedding_function.py`**: A função de embedding do nosso modelo usando nomic-embed-text devivo a limitada capacidade de hardware.
* **`populate_database.py`**: Nossa função que carrega novos arquivos para o banco de dados vetorial ou reseta usando --reset
* **`query_data.py`**: Nosso "chat" com nosso modelo, usando python query_data.py "-Qualquer pergunta envolvendo os pdfs no banco de dados-".
* **`test_rag.py`**: A nossa forma de avaliar nosso modelo usando pytest teste_rag.py.
* **`data\_qualquer_arquivo_PDF_.pdf`**: O código-fonte da nossa aplicação web (Frontend e integração com o modelo), escrito usando o Streamlit.
* **`requirements.txt`**: Lista de todas as bibliotecas e dependências do Python necessárias para rodar o projeto.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **LLM & Embeddings:** Ollama (Modelos: Mistral e nomic-embed-text)
* **Orquestração RAG:** Langchain
* **Banco de Dados Vetorial:** ChromaDB
* **Testes & Validação:** Pytest

---

## 💻 Como rodar o projeto localmente

Se você quiser clonar este projeto e rodá-lo na sua própria máquina, siga os passos abaixo:

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/linekerdasaeve/local-rag-pdf.git](https://github.com/linekerdasaeve/local-rag-pdf.git)
   cd local-rag-pdf
   
2. **Crie um ambiente virtual (Opcional, mas recomendado):**
	```bash
	python -m venv venv
	source venv/bin/activate  # No Windows use: venv\Scripts\activate
	
3. **Instale as dependências:**
	```bash
	pip install -r requirements.txt

4. **Adicione os seus documentos: :**
Cole os arquivos em PDF que você deseja analisar dentro da pasta data/. Exemplo: data/meu_documento.pdf
	
5. ** Rode o arquivo populate_database.py que alimenta o banco de dados vetorial, ou limpe usando --reset**
	```bash
	python populate_database.py
	
(Dica: Se precisar atualizar os documentos do zero, limpe o banco rodando python populate_database.py --reset)
	
6. ** Faça suas perguntas ao modelo:**
	```bash
	python query_data.py "Quantas cartas de imagem vêm no jogo Dixit?"

7. ** Avalie a integridade do modelo (Opcional):**
Para garantir que o RAG está respondendo corretamente, você pode alterar os casos de teste no arquivo test_rag.py com perguntas e respostas esperadas baseadas nos seus PDFs, e então rodar:

	```bash
	pytest test_rag.py

