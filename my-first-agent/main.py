"""Life Coach Agent — Streamlit + OpenAI Agents SDK (Agent + Runner) + 웹 검색.

실행: uv run streamlit run main.py
"""

import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from agents import Agent, Runner, WebSearchTool
from agents.items import ToolCallItem

# --- 환경 변수 로드 ---
load_dotenv(find_dotenv(usecwd=True))

MODEL = "gpt-4o-mini"  # 웹 검색(web_search) 호스티드 툴을 지원하는 모델

# --- 라이프 코치 페르소나 ---
INSTRUCTIONS = """
당신은 따뜻하고 힘이 되어주는 '라이프 코치'입니다.
사용자가 더 나은 습관을 만들고, 동기를 얻고, 스스로 성장하도록 돕는 것이 목표입니다.

행동 규칙:
1. 항상 격려하고 공감하는 태도로, 한국어로 자연스럽게 대화하세요.
2. 동기부여, 자기 계발 팁, 습관 형성 등 구체적이고 검증된 조언이 필요하면
   web_search 도구로 최신 정보를 검색한 뒤, 그 내용을 바탕으로 답하세요.
3. 조언은 두루뭉술한 말 대신 바로 실천할 수 있는 단계로 정리해서 주세요.
4. 이전 대화 내용을 기억하고, 사용자의 목표와 상황에 맞춰 이어서 코칭하세요.
5. 답변 끝에는 작은 응원 한마디나 다음에 시도해 볼 만한 행동을 제안하세요.
""".strip()


@st.cache_resource
def get_agent():
    """에이전트는 한 번만 생성해서 재사용합니다 (Streamlit 리런마다 재생성 방지)."""
    return Agent(
        name="Life Coach",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[WebSearchTool()],  # OpenAI 호스티드 웹 검색 도구
    )


def extract_search_queries(result):
    """이번 턴에서 에이전트가 실제로 수행한 웹 검색어를 뽑아냅니다 (UI 표시용)."""
    queries = []
    for item in result.new_items:
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        action = getattr(raw, "action", None)
        # 호스티드 웹 검색의 검색어는 action.query 에 들어 있습니다.
        query = getattr(action, "query", None)
        if query:
            queries.append(query)
    return queries


# --- 페이지 설정 ---
st.set_page_config(page_title="Life Coach Agent", page_icon="🌱")
st.title("🌱 Life Coach Agent")
st.caption("동기부여 · 자기 계발 · 습관 형성을 도와주는 AI 라이프 코치")

with st.sidebar:
    st.header("설정")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.display_messages = []
        st.session_state.agent_input = []
        st.rerun()

# OPENAI_API_KEY 확인
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")
    st.stop()

# --- 세션 메모리 초기화 ---
# display_messages: 화면에 그릴 메시지(role, content)
# agent_input: 에이전트에 매 턴 전달하는 전체 대화 기록(= 메모리)
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "agent_input" not in st.session_state:
    st.session_state.agent_input = []

# --- 지난 대화 다시 그리기 ---
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("searches"):
            st.caption("🔎 웹 검색: " + ", ".join(msg["searches"]))
        st.markdown(msg["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("무엇을 도와드릴까요? (예: 아침에 일찍 일어나고 싶어요)"):
    # 1) 사용자 메시지 화면에 표시 + 저장
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) 기존 대화 기록에 새 사용자 메시지를 더해 에이전트 실행
    agent = get_agent()
    next_input = st.session_state.agent_input + [{"role": "user", "content": prompt}]

    with st.chat_message("assistant"):
        with st.spinner("코치가 생각하고 있어요..."):
            try:
                result = Runner.run_sync(agent, next_input)
                answer = result.final_output
                searches = extract_search_queries(result)
                # 다음 턴을 위해 전체 대화 기록(메모리)을 갱신
                st.session_state.agent_input = result.to_input_list()
            except Exception as e:  # noqa: BLE001
                answer = f"죄송해요, 답변 중 오류가 발생했어요: {e}"
                searches = []

        if searches:
            st.caption("🔎 웹 검색: " + ", ".join(searches))
        st.markdown(answer)

    # 3) 코치 응답 저장
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "searches": searches}
    )
