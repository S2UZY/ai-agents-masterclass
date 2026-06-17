"""Life Coach Agent — Streamlit + OpenAI Agents SDK (Agent + Runner).

기능:
  - 웹 검색(WebSearchTool): 동기부여 · 자기계발 · 습관 형성 최신 정보 검색
  - 파일 검색(FileSearchTool): 개인 목표 문서 / 성장 일기를 검색해 맞춤 코칭
  - 세션 메모리: 대화 기록을 기억

실행: uv run streamlit run main.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from agents import Agent, FileSearchTool, Runner, WebSearchTool
from agents.items import ToolCallItem

# --- 환경 변수 로드 ---
load_dotenv(find_dotenv(usecwd=True))

MODEL = "gpt-4o-mini"  # 웹/파일 검색 호스티드 툴을 지원하는 모델
GOALS_FILE = Path(__file__).parent / "goals.txt"  # 기본 개인 목표 문서
VS_ID_FILE = Path(__file__).parent / ".vector_store_id"  # 벡터 스토어 id 저장(재사용)
VECTOR_STORE_NAME = "life-coach-goals"

# --- 라이프 코치 페르소나 ---
INSTRUCTIONS = """
당신은 따뜻하고 힘이 되어주는 '라이프 코치'입니다.
사용자가 더 나은 습관을 만들고, 동기를 얻고, 목표를 향해 나아가도록 돕는 것이 목표입니다.

당신에게는 두 가지 도구가 있습니다:
- file_search: 사용자의 '개인 목표 문서'와 '성장 일기'를 검색합니다.
- web_search: 동기부여 · 자기계발 · 습관 형성에 대한 최신 정보를 검색합니다.

행동 규칙:
1. 사용자의 목표·진행 상황·과거 기록과 관련된 질문이면 먼저 file_search 로
   개인 문서를 찾아 사용자의 구체적인 목표와 지난 기록을 확인하세요.
2. 검증된 방법이나 최신 팁이 필요하면 web_search 로 검색하세요.
3. 두 도구를 결합해, '사용자의 목표 + 과거 기록'에 맞춘 개인화된 조언을 주세요.
   (예: 목표 문서에서 '주 3회 운동'을 확인 → 웹에서 루틴 유지법 검색 → 둘을 엮어 제안)
4. 조언은 두루뭉술한 말 대신 바로 실천할 수 있는 단계로 정리해 주세요.
5. 시간에 따른 진행 상황을 짚어주고(예: "3월보다 러닝 거리가 늘었네요"),
   작은 성취는 꼭 축하해 주세요.
6. 항상 격려하고 공감하는 태도로, 한국어로 자연스럽게 대화하세요.
""".strip()


@st.cache_resource(show_spinner="목표 문서를 준비하고 있어요...")
def setup_vector_store():
    """개인 목표 문서를 담는 벡터 스토어를 만들거나 재사용합니다.

    - .vector_store_id 파일이 있고 유효하면 그대로 재사용
    - 없으면 새로 만들고 goals.txt 를 업로드한 뒤 id 를 저장
    """
    client = OpenAI()

    # 1) 기존 벡터 스토어 재사용 시도
    if VS_ID_FILE.exists():
        vs_id = VS_ID_FILE.read_text().strip()
        try:
            client.vector_stores.retrieve(vs_id)
            return client, vs_id
        except Exception:  # noqa: BLE001 (삭제됐거나 잘못된 id면 새로 생성)
            pass

    # 2) 새 벡터 스토어 생성 + 기본 목표 문서 업로드
    vs = client.vector_stores.create(name=VECTOR_STORE_NAME)
    if GOALS_FILE.exists():
        with GOALS_FILE.open("rb") as f:
            client.vector_stores.files.upload_and_poll(vector_store_id=vs.id, file=f)
    VS_ID_FILE.write_text(vs.id)
    return client, vs.id


def ingest_uploaded_file(client, vs_id, uploaded_file):
    """사이드바에서 업로드한 문서(PDF/TXT)를 벡터 스토어에 추가합니다."""
    client.vector_stores.files.upload_and_poll(
        vector_store_id=vs_id,
        file=(uploaded_file.name, uploaded_file.getvalue()),
    )


def list_ingested_files(client, vs_id):
    """벡터 스토어에 들어 있는 문서 파일명을 가져옵니다 (사이드바 표시용)."""
    names = []
    try:
        for vf in client.vector_stores.files.list(vector_store_id=vs_id).data:
            meta = client.files.retrieve(vf.id)
            names.append(getattr(meta, "filename", vf.id))
    except Exception:  # noqa: BLE001
        pass
    return names


@st.cache_resource
def get_agent(vs_id):
    """에이전트는 한 번만 생성해 재사용합니다 (Streamlit 리런마다 재생성 방지)."""
    return Agent(
        name="Life Coach",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[
            FileSearchTool(vector_store_ids=[vs_id]),  # 개인 목표 문서 검색
            WebSearchTool(),  # 웹 검색
        ],
    )


def extract_tool_activity(result):
    """이번 턴에 수행한 검색 활동을 사람이 읽을 수 있는 문자열로 뽑아냅니다."""
    activity = []
    for item in result.new_items:
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        kind = getattr(raw, "type", None)
        if kind == "web_search_call":
            query = getattr(getattr(raw, "action", None), "query", None)
            activity.append(f"🔎 웹 검색: {query}" if query else "🔎 웹 검색")
        elif kind == "file_search_call":
            queries = getattr(raw, "queries", None) or []
            label = ", ".join(queries) if queries else ""
            activity.append(f"📄 목표 문서 검색: {label}" if label else "📄 목표 문서 검색")
    return activity


# --- 페이지 설정 ---
st.set_page_config(page_title="Life Coach Agent", page_icon="🌱")
st.title("🌱 Life Coach Agent")
st.caption("개인 목표를 기억하고 · 진행 상황을 추적하며 · 맞춤 조언을 주는 AI 라이프 코치")

# OPENAI_API_KEY 확인
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")
    st.stop()

# --- 벡터 스토어 & 에이전트 준비 ---
client, vs_id = setup_vector_store()
agent = get_agent(vs_id)

# --- 사이드바: 문서 업로드 / 관리 ---
with st.sidebar:
    st.header("📁 내 목표 문서")
    uploaded = st.file_uploader(
        "목표·일기 문서 업로드 (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploaded and st.button("문서 추가하기", use_container_width=True):
        with st.spinner("문서를 색인하고 있어요..."):
            for f in uploaded:
                ingest_uploaded_file(client, vs_id, f)
        st.success(f"{len(uploaded)}개 문서를 추가했어요!")

    st.divider()
    st.caption("현재 코치가 참고하는 문서:")
    for name in list_ingested_files(client, vs_id):
        st.markdown(f"- {name}")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.display_messages = []
        st.session_state.agent_input = []
        st.rerun()

# --- 세션 메모리 초기화 ---
# display_messages: 화면에 그릴 메시지(role, content, 검색활동)
# agent_input: 에이전트에 매 턴 전달하는 전체 대화 기록(= 메모리)
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "agent_input" not in st.session_state:
    st.session_state.agent_input = []

# --- 지난 대화 다시 그리기 ---
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        for line in msg.get("activity", []):
            st.caption(line)
        st.markdown(msg["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("무엇을 도와드릴까요? (예: 내 운동 목표 잘 되어가고 있어?)"):
    # 1) 사용자 메시지 표시 + 저장
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) 기존 대화 기록에 새 메시지를 더해 에이전트 실행
    next_input = st.session_state.agent_input + [{"role": "user", "content": prompt}]

    with st.chat_message("assistant"):
        with st.spinner("코치가 목표를 살펴보고 있어요..."):
            try:
                result = Runner.run_sync(agent, next_input)
                answer = result.final_output
                activity = extract_tool_activity(result)
                # 다음 턴을 위해 전체 대화 기록(메모리) 갱신
                st.session_state.agent_input = result.to_input_list()
            except Exception as e:  # noqa: BLE001
                answer = f"죄송해요, 답변 중 오류가 발생했어요: {e}"
                activity = []

        for line in activity:
            st.caption(line)
        st.markdown(answer)

    # 3) 코치 응답 저장
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "activity": activity}
    )
