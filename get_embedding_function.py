import warnings
warnings.filterwarnings("ignore")

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    # Inicializa e retorna a função de embedding para o pipeline do RAG.
    # 
    # Contexto da escolha:
    # Estamos rodando o RAG localmente. Devido a limitações de hardware da máquina atual,
    # optamos pelo modelo 'nomic-embed-text' via Ollama, que é mais leve.
    # 
    # Alternativas e Escalabilidade:
    # - Locais: Existem modelos locais mais robustos (como o 'mistral', exemplificado abaixo).
    # - Nuvem: Para maior qualidade e uso em produção, recomenda-se provedores online 
    #   (AWS, OpenAI, etc.), que requerem configuração de API Keys.
    # 
    # Documentação para conectar outros provedores no LangChain:
    # URL: https://python.langchain.com/docs/integrations/text_embedding/
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# Exemplo de configuração alternativa utilizando o modelo Mistral localmente:
'''def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mistral")
    return embeddings'''