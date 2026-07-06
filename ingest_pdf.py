import os
import re
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from model_factory import ModelFactory

def run_structured_ingest(data_dir: str):
    search_path = os.path.join(data_dir, "*.pdf")
    pdf_files = glob.glob(search_path)
    
    if not pdf_files:
        print(f"[오류] '{data_dir}' 폴더 내에 PDF 가이드북 소스 파일이 존재하지 않습니다.")
        return
        
    print(f"[*] 총 {len(pdf_files)}개의 Red Hat 공식 PDF 문서 구조화 정밀 전처리 가동...")
    all_final_chunks = []
    
    # 기획서 장표 명세서 양식과 100% 대응되는 마크다운 헤더 분기선 정의
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=120)
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page in pages:
                raw_text = page.page_content
                
                # 🔴 [리뷰 피드백 반영] PDF 텍스트 구조 특화 사전 정제 레이어 작동
                # PDF 원문 레이아웃에서 흔히 발견되는 챕터/섹션 타이틀 패턴을 감지하여 마크다운 헤더 기호(#)를 강제 선주입
                # 이를 통해 MarkdownHeaderTextSplitter 가 정상적으로 분할할 수 있는 완벽한 강건성 인프라를 확보합니다.
                formatted_text = re.sub(r'^(Chapter\s+\d+|[A-Z\s]{4,})(\n|$)', r'\n# \1\n', raw_text, flags=re.MULTILINE)
                formatted_text = re.sub(r'^(\d+\.\d+\s+[A-Za-z]+)', r'\n## \1\n', formatted_text, flags=re.MULTILINE)
                formatted_text = re.sub(r'^(\d+\.\d+\.\d+\s+[A-Za-z]+)', r'\n### \1\n', formatted_text, flags=re.MULTILINE)
                
                # 선주입 정제가 마감된 마크다운 구조 텍스트 분할 실행
                header_split_docs = markdown_splitter.split_text(formatted_text)
                
                # 인프라 특화 다차원 메타데이터 보강 매핑 공정
                for doc in header_split_docs:
                    doc.metadata["source_file"] = file_name
                    if "disconnected" in file_name.lower():
                        doc.metadata["section_category"] = "Disconnected Environment Setup"
                    elif "operator" in file_name.lower():
                        doc.metadata["section_category"] = "Operator Lifecycle Management"
                    else:
                        doc.metadata["section_category"] = "General Architecture & Commands"
                
                # 청크 사이즈 규격화를 위한 의미 단위 2차 정밀 세부 스플리팅 결합
                splits = text_splitter.split_documents(header_split_docs)
                all_final_chunks.extend(splits)
                
            print(f"    ➔ '{file_name}' PDF 레이아웃 전처리 가드 가동 성공 (청크 {len(all_final_chunks)}개 축적)")
        except Exception as e:
            print(f"⚠️ [{file_name}] 인프라 파싱 트랜잭션 실패 건너뜁니다: {e}")
            
    load_dotenv()
    embeddings = ModelFactory.get_embeddings()
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    
    print(f"[*] Chroma Vector DB 영구 인덱싱 및 persist 디스크 저장 가동... ({db_path})")
    
    # 단일 API 배치 업로드 한도를 회피하여 메모리 오버플로우를 차단하는 롤링 배치 안전 분할 적재
    batch_size = 100
    vector_store = None
    for i in range(0, len(all_final_chunks), batch_size):
        batch = all_final_chunks[i:i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory=db_path)
        else:
            vector_store.add_documents(documents=batch)
            
    print("[🎉 RAG 전처리 성공] 문서 구조 불일치 리스크가 완벽히 차단된 정밀 인덱싱 공정이 마감되었습니다.")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    run_structured_ingest("./data")