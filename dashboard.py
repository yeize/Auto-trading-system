# dashboard.py (라이브 대시보드 전용 최종 버전)

import streamlit as st
import json
import time
import os
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh # 자동 새로고침

# --- Constants & Config ---
STATUS_FILE = 'status.json'
REFRESH_INTERVAL_MS = 5000 # 자동 새로고침 간격 (밀리초)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Trading Bot Live Dashboard",
    page_icon="🤖",
    layout="wide"
    # initial_sidebar_state 는 멀티페이지에서 자동으로 관리됨
)

# --- Initialize State ---
if 'signal_log' not in st.session_state: st.session_state.signal_log = []

# --- Utility Functions ---
def load_status():
    default_status = {"bot_status": "UNKNOWN"};
    if not os.path.exists(STATUS_FILE): return "봇 대기 중", {}
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f: content = f.read()
        if not content: return "상태 파일 비어있음", {}
        status = json.loads(content); return status.get("bot_status", "로드됨"), status
    except Exception as e: return f"상태 파일 오류: {e}", {}

def append_signal_log(signal, timestamp):
    display_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    signal_map = {"ENTER_LONG": "▲ LONG 진입", "ENTER_SHORT": "▼ SHORT 진입", "EXIT_LONG": "▽ LONG 종료", "EXIT_SHORT": "△ SHORT 종료", "NONE": "없음"}
    signal_entry = f"[{display_time}] {signal_map.get(signal, signal)}"
    if signal != "NONE" and (not st.session_state.signal_log or st.session_state.signal_log[-1] != signal_entry):
        st.session_state.signal_log.append(signal_entry)
        if len(st.session_state.signal_log) > 10: st.session_state.signal_log.pop(0)

def update_live_dashboard_data():
    bot_status, data = load_status()
    default_data = {"symbol": "-", "current_price": 0.0, "base_price": 0.0, "unrealized_pnl_percent": 0.0,"available_balance": 0.0, "current_position": "NONE", "entry_price": None, "last_signal_display": "NONE", "timestamp": time.time(), "log_messages": []}
    if data: default_data.update(data)
    # 로그 추가 로직은 여기서 계속 관리
    append_signal_log(default_data.get('last_signal_display', 'NONE'), default_data.get('timestamp', time.time()))
    return bot_status, default_data

# --- 앱 실행 ---

# 자동 새로고침 실행
# 참고: st.session_state.optimization_running은 다른 페이지에서 설정되므로 여기서는 직접 사용하지 않음
#       대신, 이 페이지는 항상 자동 새로고침되도록 설정
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="live_dashboard_refresh")

# --- 라이브 대시보드 UI 구성 ---
st.title("🤖 트레이딩 봇 대시보드 (Live)")

# 데이터 로드
bot_status, data = update_live_dashboard_data()

# 메트릭 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("봇 상태", bot_status); col1.metric("심볼", data.get('symbol', '-'))
col2.metric("현재가", f"{data.get('current_price', 0.0):.4f}") # XRP 가격 표시 위해 소수점 늘림
col2.metric("기준가", f"{data.get('base_price', 0.0):.4f}") # XRP 가격 표시 위해 소수점 늘림
position_display = data.get('current_position', 'NONE'); entry_price_display = f"{data.get('entry_price', 'N/A'):.4f}" if isinstance(data.get('entry_price'), (int, float)) else "N/A"; pnl_display = f"{data.get('unrealized_pnl_percent', 0.0):.2f}%"
col3.metric("포지션", position_display); col3.metric("진입 가격", entry_price_display); col3.metric("미실현 손익 (%)", pnl_display)
col4.metric("잔고 (USDT)", f"{data.get('available_balance', 0.0):.2f}"); last_update_ts = data.get('timestamp', time.time()); col4.metric("최종 업데이트", datetime.fromtimestamp(last_update_ts).strftime('%Y-%m-%d %H:%M:%S'))

# 최근 신호 로그 표시
st.subheader("📊 최근 신호 로그 (Live)")
sig_log_placeholder = st.empty()
if st.session_state.signal_log: log_df = pd.DataFrame(st.session_state.signal_log, columns=["신호 로그"]); sig_log_placeholder.dataframe(log_df.iloc[::-1], use_container_width=True, height=min(35 * len(st.session_state.signal_log) + 38, 400))
else: sig_log_placeholder.info("아직 기록된 신호가 없습니다.")

# 실시간 봇 로그 표시
st.subheader("📜 실시간 봇 로그")
realtime_log_placeholder = st.empty()
log_messages = data.get("log_messages", ["로그를 기다리는 중..."]); log_display_text = "\n".join(log_messages[::-1])
realtime_log_placeholder.text_area("Log Output:", value=log_display_text, height=300, key="log_display_area", disabled=True)