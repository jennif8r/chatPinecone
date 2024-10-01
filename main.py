from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import langchain
from Agents.ti_agent import TiAgent
from Agents.rh_agent import RhAgent
from Agents.classifier_agent import ClassifierAgent

# Carregar variáveis de ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuração inicial
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

# Configurar o modelo LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, transport='rest', temperatura=0.1, max_tokens=6)

classifier_agent = ClassifierAgent(llm)
ti_agent = TiAgent()
rh_agent = RhAgent()


def main(question):

 
    category, classification_obj = classifier_agent.classify_question(question)
    category = category.lower()

    if "ti" in category:
        return ti_agent.handle_ti_question(), classification_obj
    elif "rh" in category:
        return rh_agent.handle_rh_question(), classification_obj
    else:
        return "Categoria não encontrada. Por favor, faça perguntas relacionadas às áreas reconhecidas.", classification_obj


# Exemplo de uso
if __name__ == "__main__":
    user_question = "Como é calculado o período de férias de um funcionário?"
    response, classification_obj = main(user_question)
    print(response)


