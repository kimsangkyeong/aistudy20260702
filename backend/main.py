import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from backend.agent_graph import compiled_graph

load_dotenv()
app = FastAPI(title="OCP-Ops Backend Core Engine", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="엔지니어 조치 요구사항 질의 텍스트")
    thread_id: str = Field("default_ops_thread", description="맥락 연속성 유지를 위한 유저 스레드 ID")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="공백 질의는 적재 연산 처리가 불가능합니다.")
        
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        inputs = {"messages": [("user", request.message.strip())]}
        
        # 🔴 피드백 반영: 원시적 파싱을 철폐하고 LangGraph State에서 검증 완료된 Pydantic 구조화 딕셔너리 직접 매핑 추출
        output_state = compiled_graph.invoke(inputs, config=config)
        final_output = output_state.get("final_structured_output")
        
        if not final_output:
            raise ValueError("그래프 아웃풋 스키마 조립 실패")
            
        return final_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 다자간 순환 제어 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)