# backend/main.py
from fastapi import FastAPI, HTTPException, status
from backend.schemas import ChatRequest, OCPResponseSchema
from backend.agent_graph import compiled_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="OCP-Ops Agent Core Engine", version="2.0.0")

@app.post("/api/chat", response_model=OCPResponseSchema, status_code=status.HTTP_200_OK)
async def handle_chat_endpoint(payload: ChatRequest):
    # 1. 다양한 입력 상황에 대응하기 위한 1차 사용자 인풋 데이터 무결성 검증
    clean_message = payload.message.strip() if payload.message else ""
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="입력 필드가 비어 있습니다. OpenShift 4.20 운영 관련 질의를 입력해 주세요."
        )
        
    try:
        config = {"configurable": {"thread_id": payload.thread_id}}
        # 초기 루프 상태 초기화 포함 주입
        initial_state = {
            "messages": [HumanMessage(content=clean_message)],
            "validation_loop_count": 0
        }
        
        # 2. LangGraph 멀티 에이전트 다자간 교차 검증 워크플로우 엔진 가동
        final_state = compiled_graph.invoke(initial_state, config=config)
        
        # 3. 불안정한 문자열 파싱(replace)을 완전히 제거하고, Validator가 최종 상태 저장소에 적재한 객체 직접 추출
        structured_data = final_state.get("final_structured_output")
        
        if not structured_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="에이전트 협업 검증 레이어 내에서 유효한 응답 구조를 형성하지 못했습니다."
            )
            
        # Pydantic 구조화 출력을 보장하며 즉시 딕셔너리 포맷 리턴
        return structured_data
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[백엔드 시스템 크리티컬 런타임 예외 리포트] : {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"인프라 백엔드 컴파일러 붕괴: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)