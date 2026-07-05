import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from model_factory import ModelFactory

def run_structured_ingest(data_dir: str):
    search_path = os.path.join(data_dir, "*.pdf")
    pdf_files = glob.glob(search_path)
    
    if not pdf_files:
        print(f"[오류] '{data_dir}' 폴더 내에 PDF 파일이 없습니다.")
        return

    print(f"[*] 총 {len(pdf_files)}개의 문서 분석 및 고도화 섹션 청킹 가동...")
    all_sections = []
    
    # 세부 인프라 명세 분할을 위해 separators 최적화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " "]
    )

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 문서 구조화를 위해 페이지 콘텐츠 상단 메타데이터 매핑 공정 보완
            for page in pages:
                page.metadata["source_file"] = file_name
                if "disconnected" in file_name.lower():
                    page.metadata["section_category"] = "Disconnected Environment Setup"
                elif "operator" in file_name.lower():
                    page.metadata["section_category"] = "Operator Lifecycle Management"
                else:
                    page.metadata["section_category"] = "General Architecture & Commands"

            chunks = text_splitter.split_documents(pages)
            all_sections.extend(chunks)
            print(f"    ➔ '{file_name}' 전처리 완료: 청크 {len(chunks)}개 변환")
        except Exception as e:
            print(f"⚠️ [{file_name}] 파싱 실패 건너뜁니다: {e}")

    load_dotenv()
    embeddings = ModelFactory.get_embeddings()
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))

    print(f"[*] Chroma Vector DB 적재 및 영구 인덱싱 진행 중... ({db_path})")
    
    # 토큰 한도 초과 방지 분할 배치 루프
    batch_size = 100
    vector_store = None
    for i in range(0, len(all_sections), batch_size):
        batch = all_sections[i:i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory=db_path)
        else:
            vector_store.add_documents(documents=batch)
            
    print("[🎉 성공] 문서 구조화 기반 RAG 인덱싱 공정이 완벽히 마감되었습니다.")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    run_structured_ingest("./data")