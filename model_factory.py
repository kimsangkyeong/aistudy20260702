import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class ModelFactory:
    @staticmethod
    def get_llm():
        env_type = os.getenv("RUNNING_ENV", "GEMINI").upper()
        
        if env_type == "AZURE":
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
                openai_api_key=os.getenv("AOAI_API_KEY"),
                azure_endpoint=os.getenv("AOAI_ENDPOINT"),
                api_version=os.getenv("AOAI_API_VERSION"),
                temperature=0
            )
        else:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )

    @staticmethod
    def get_embeddings():
        env_type = os.getenv("RUNNING_ENV", "GEMINI").upper()
        
        if env_type == "AZURE":
            return AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
                openai_api_key=os.getenv("AOAI_API_KEY"),
                azure_endpoint=os.getenv("AOAI_ENDPOINT"),
                api_version=os.getenv("AOAI_API_VERSION"),
            )
        else:
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )