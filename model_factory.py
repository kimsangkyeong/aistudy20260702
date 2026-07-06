import os
from dotenv import load_dotenv

# 파일 최상단에서 환경변수 로딩 수행
load_dotenv()

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class ModelFactory:
    @staticmethod
    def get_llm():
        env_type = os.getenv("RUNNING_ENV", "GEMINI").upper()
        
        if env_type == "AZURE":
            # 🔥 [교정] 엔지니어님이 설정하신 'AOAI_ENDPOINT' 명칭으로 정확히 매핑합니다.
            endpoint = os.getenv("AOAI_ENDPOINT")
            api_key = os.getenv("AOAI_API_KEY")
            
            if not endpoint or not api_key:
                print("[⚠️ 경고] .env 파일에서 AOAI_ENDPOINT 또는 AOAI_API_KEY를 읽지 못했습니다.")
                
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
                openai_api_key=api_key,
                azure_endpoint=endpoint,
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
            # 🔥 [교정] 임베딩 측 엔드포인트도 'AOAI_ENDPOINT' 명칭으로 수정합니다.
            endpoint = os.getenv("AOAI_ENDPOINT")
            api_key = os.getenv("AOAI_API_KEY")
            
            # LangChain 표준 프레임워크 호환용 글로벌 스코프 강제 주입
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
                google_api_key=os.getenv("GEMINI_API_KEY")
            )