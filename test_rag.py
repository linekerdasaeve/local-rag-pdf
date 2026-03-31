import warnings
warnings.filterwarnings("ignore")
from query_data import query_rag
from langchain_community.llms.ollama import Ollama


# O código abaixo serve para avaliar nosso modelo
# Ele utiliza uma técnica chamada "LLM-as-a-judge", onde um modelo de linguagem
# (neste caso, o Mistral) é usado para avaliar as respostas do nosso RAG.

# Prompt de avaliação: Instruções estritas para o modelo avaliador comparar
# a resposta esperada com a resposta real gerada pelo RAG e retornar apenas 'true' ou 'false'.
EVAL_PROMPT = """
Resposta Esperada: {expected_response}
Resposta Dada: {actual_response}
---
(Responda com 'verdadeiro' ou 'falso') A resposda dada corresponde a resposta esperada? 
"""

# --- Casos de Teste ---
# Estas funções definem as perguntas e as respostas exatas que esperamos.
# Geralmente são executadas usando frameworks de teste como o 'pytest'.

def teste_cartas_de_imagem():
    # Testa se o RAG sabe a quantidade total de cartas no jogo Dixit.
    assert query_and_validate(
        question="Quantas cartas de imagem vêm no jogo Dixit? (responda apenas com números)",
        expected_response="84 cartas de imagem",
    )


def teste_cartas_final_de_turno():
    # Testa se o RAG conhece as regras de compra de cartas no final do turno.
    assert query_and_validate(
        question="Na partida com três participantes, quantas cartas eu devo ter na minha mão ao final do turno e após comprar as cartas? (responda apenas com números)",
        expected_response="7 cartas",
    )


# --- Função Principal de Validação ---
def query_and_validate(question: str, expected_response: str):
    # Passo 1: Faz a pergunta ao nosso sistema RAG e guarda a resposta real.
    response_text = query_rag(question)
    
    # Passo 2: Preenche o template de avaliação com a resposta esperada e a obtida.
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Passo 3: Inicializa o modelo local (Mistral) que servirá como juiz.
    model = Ollama(model="mistral")
    
    # Passo 4: Envia o prompt de avaliação para o juiz e limpa a resposta (remove espaços e converte para minúsculo).
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    # Imprime o prompt gerado no terminal para fins de depuração (debug).
    print(prompt)

    # Passo 5: Analisa o veredito do LLM juiz e aplica cores no terminal para facilitar a visualização.
    if "verdadeiro" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct. (Verde para sucesso)
        print("\033[92m" + f"Resposta: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "falso" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect. (Vermelho para falha)
        print("\033[91m" + f"Resposta: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        # Tratamento de erro: Caso o LLM responda algo fora do padrão 'true' ou 'false'.
        raise ValueError(
            f"Avaliação com resultado inválido. Não foi possível determinar se é 'verdadeiro' ou 'falso'."            
        )