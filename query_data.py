import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import warnings
warnings.filterwarnings("ignore")

# Caminho para o diretório onde o nosso banco de dados vetorial está salvo localmente.
CHROMA_PATH = "chroma"

# Template do Prompt: Esta é uma parte crucial do RAG. 
# Estamos instruindo o modelo a responder estritamente com base no contexto fornecido,
# evitando que ele tenha "alucinações" ou use conhecimentos prévios fora da nossa base.
PROMPT_TEMPLATE = """
Responda a pergunta abaixo com base no contexto a seguir:
{context}

---
Responda a pergunta com base no contexto acima:
{question}
"""

def main():
    # Cria a interface de linha de comando
    # Isso permite rodar o script diretamente no terminal passando a pergunta.
    # Exemplo: python query_data.py "Quantas cartas tem no Dixit?"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    # Chama a função principal passando a pergunta do usuário
    query_rag(query_text)

def query_rag(query_text: str):
    # --- PASSO 1: Preparação do Banco de Dados ---
    # Carrega a função de embedding (a mesma usada para criar o banco)
    # e conecta ao ChromaDB existente no diretório especificado.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # --- PASSO 2: Recuperação (Retrieval) ---
    # Busca no banco vetorial os trechos mais parecidos com a pergunta do usuário.
    # O parâmetro k=5 indica que queremos os 5 resultados mais relevantes.
    results = db.similarity_search_with_score(query_text, k=5)

    # --- PASSO 3: Aumento de Contexto (Augmentation) ---
    # Junta o texto de todos os 5 documentos encontrados, separando-os por "---"
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Preenche o nosso template com o contexto recuperado e a pergunta original
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt) # Útil descomentar para debugar o que está sendo enviado ao LLM

    # --- PASSO 4: Geração (Generation) ---
    # Inicializa o LLM local (Mistral) via Ollama e envia o prompt completo.
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # --- PASSO 5: Formatação da Saída ---
    # Coleta os IDs ou nomes dos arquivos fonte (salvos nos metadados durante a criação do banco)
    # para sabermos exatamente de onde a informação foi tirada.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Resposta: {response_text}\nSources: {sources}"
    
    # Exibe no console a resposta final do modelo junto com as fontes utilizadas
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
