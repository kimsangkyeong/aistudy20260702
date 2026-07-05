from pydantic import BaseModel, Field
from typing import List

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "ocp_default_thread"
 
class OCPResponseSchema(BaseModel):
    summary: str = Field(description="조치 결과에 대한 핵심 요약")
    steps: List[str] = Field(description="폐쇄망 환경에서 수행해야 하는 단계별 가이드")
    code_block: str = Field(description="터미널에 즉시 실행 가능한 oc 명령어 또는 YAML")
    references: List[str] = Field(description="참고한 Red Hat 공식 문서 출처 및 링크")