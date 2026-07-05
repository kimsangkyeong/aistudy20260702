# model_factory.py (글로벌 규격 완벽 정제 버전)
import os
from dotenv import load_dotenv

# 파일 최상단에서 시스템 인프라 환경 변수 확실하게 로드
load_dotenv()

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class ModelFactory:
    @staticmethod
    def get_llm():
        env_type = os.getenv("RUNNING_ENV", "GEMINI").upper()
        
        if env_type == "AZURE":
            endpoint = os.getenv("AOAI_ENDPOINT")
            api_key = os.getenv("AOAI_API_KEY")
            
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                api_version=os.getenv("AOAI_API_VERSION"),
                temperature=0
            )
        else:
            # api_version="v1" 지정을 통해 하위 드라이버의 v1beta 엉뚱한 변환을 막고 gemini-1.5-pro와 정석 연동
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                api_version="v1",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )

    @staticmethod
    def get_embeddings():
        env_type = os.getenv("RUNNING_ENV", "GEMINI").upper()
        
        if env_type == "AZURE":
            endpoint = os.getenv("AOAI_ENDPOINT")
            api_key = os.getenv("AOAI_API_KEY")
            
            if endpoint:
                os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
            if api_key:
                os.environ["AZURE_OPENAI_API_KEY"] = api_key
                
            return AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"),
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                api_version=os.getenv("AOAI_API_VERSION"),
            )
        else:
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                api_version="v1",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )