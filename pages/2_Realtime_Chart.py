# pages/2_📈_Realtime_Chart.py (현재가 표시 추가)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from streamlit_autorefresh import st_autorefresh
import time
import pandas_ta as ta
import traceback
import numpy as np

# --- 설정 ---
REFRESH_INTERVAL_MS = 5000
DEFAULT_SYMBOL = "XRPUSDT"
DEFAULT_INTERVAL_KEY = "1h"
DEFAULT_LIMIT = 150

POPULAR_FUTURES_SYMBOLS = sorted([
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "TRXUSDT", "LTCUSDT", "ETCUSDT",
    "BCHUSDT", "NEARUSDT", "OPUSDT", "APTUSDT", "ARBUSDT", "FILUSDT", "ATOMUSDT",
    "UNIUSDT", "XLMUSDT", "APEUSDT", "EOSUSDT", "AAVEUSDT", "FTMUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "1000PEPEUSDT", "1000SHIBUSDT"
])
INTERVAL_OPTIONS = {
    "1m": Client.KLINE_INTERVAL_1MINUTE, "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE, "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR, "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY, "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH
}
INDICATOR_OPTIONS = ["SMA 5", "SMA 10", "SMA 20", "SMA 60", "SMA 120", "Bollinger Bands (20, 2)", "MACD", "Stochastic"]
DEFAULT_INDICATORS = ["SMA 20", "SMA 60"]

# --- 페이지 설정 ---
st.set_page_config(page_title="실시간 차트", page_icon="📈", layout="wide")
st.title("📈 실시간 선물 차트")

# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=30)
def fetch_and_prepare_data(symbol, interval, limit):
    # ... (이전과 동일) ...
    try:
        client = Client("", "")
        fetch_limit = limit + 150
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=fetch_limit)
        if not klines: return None, f"{symbol} ({interval}) 데이터 로드 실패."
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms'); df.set_index('Open time', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); df = df.apply(pd.to_numeric, errors='coerce'); df.dropna(subset=df.columns, inplace=True)
        if df.empty: return None, "숫자 변환 후 유효 데이터 없음."
        df.ta.sma(length=5, append=True); df.ta.sma(length=10, append=True); df.ta.sma(length=20, append=True); df.ta.sma(length=60, append=True); df.ta.sma(length=120, append=True)
        df.ta.bbands(length=20, std=2, append=True); df.ta.macd(append=True); df.ta.stoch(append=True)
        return df.iloc[-limit:], None
    except Exception as e: return None, f"데이터 처리 중 오류: {e}\n{traceback.format_exc()}"

# --- 차트 생성 함수 ---
def plot_candlestick_with_indicators(df, symbol, interval_key, selected_indicators):
    # ... (이전과 동일) ...
    rows = 1; row_heights = [0.6]; subplot_titles = [f"{symbol} 가격 ({interval_key})"]
    indicator_rows = {}
    if "MACD" in selected_indicators: rows += 1; row_heights.append(0.2); subplot_titles.append("MACD"); indicator_rows["MACD"] = rows
    if "Stochastic" in selected_indicators: rows += 1; row_heights.append(0.2); subplot_titles.append("Stochastic"); indicator_rows["Stochastic"] = rows
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=subplot_titles, row_heights=row_heights, specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (rows - 1) )
    price_row = 1
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='캔들'), row=price_row, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(150, 150, 150, 0.5)'), row=price_row, col=1, secondary_y=True)
    sma_colors = {'SMA 5': 'rgba(255, 100, 100, 0.9)', 'SMA 10': 'rgba(255, 150, 50, 0.9)', 'SMA 20': 'rgba(200, 200, 0, 0.9)', 'SMA 60': 'rgba(100, 150, 255, 0.9)', 'SMA 120': 'rgba(180, 100, 255, 0.9)'}
    for indicator_name, color in sma_colors.items():
        if indicator_name in selected_indicators:
            sma_col = indicator_name.replace(" ", "_")
            if sma_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[sma_col], name=indicator_name, line=dict(color=color, width=1)), row=price_row, col=1, secondary_y=False)
    if "Bollinger Bands (20, 2)" in selected_indicators:
        bbu_col = 'BBU_20_2.0'; bbl_col = 'BBL_20_2.0'; bbm_col = 'BBM_20_2.0'
        if bbu_col in df.columns and bbl_col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], name='BB 상단', line=dict(color='rgba(152,251,152, 0.5)', width=1)), row=price_row, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], name='BB 하단', line=dict(color='rgba(152,251,152, 0.5)', width=1), fill='tonexty', fillcolor='rgba(152,251,152, 0.1)'), row=price_row, col=1, secondary_y=False)
        if bbm_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[bbm_col], name='BB 중간선', line=dict(color='rgba(255, 140, 0, 0.6)', width=1, dash='dash')), row=price_row, col=1, secondary_y=False)
    if "MACD" in selected_indicators:
        macd_row = indicator_rows["MACD"]; macd_col = 'MACD_12_26_9'; macds_col = 'MACDs_12_26_9'; macdh_col = 'MACDh_12_26_9'
        if macd_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[macd_col], name='MACD', line=dict(color='blue', width=1)), row=macd_row, col=1)
        if macds_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[macds_col], name='Signal', line=dict(color='red', width=1)), row=macd_row, col=1)
        if macdh_col in df.columns: fig.add_trace(go.Bar(x=df.index, y=df[macdh_col], name='Histogram', marker_color='rgba(100, 100, 100, 0.5)'), row=macd_row, col=1)
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1, zeroline=True, zerolinewidth=1, zerolinecolor='Gray')
    if "Stochastic" in selected_indicators:
        stoch_row = indicator_rows["Stochastic"]; stochk_col = 'STOCHk_14_3_3'; stochd_col = 'STOCHd_14_3_3'
        if stochk_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[stochk_col], name='%K', line=dict(color='purple', width=1)), row=stoch_row, col=1)
        if stochd_col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[stochd_col], name='%D', line=dict(color='orange', width=1)), row=stoch_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)", row=stoch_row, col=1); fig.add_hline(y=20, line_dash="dash", line_color="rgba(0, 255, 0, 0.5)", row=stoch_row, col=1)
        fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=stoch_row, col=1)
    chart_height = 450 + max(0, rows - 1) * 150
    fig.update_layout(title=f"{symbol} 실시간 차트 ({interval_key})", height=chart_height, xaxis_rangeslider_visible=False, xaxis_range=[df.index.min(), df.index.max()], yaxis=dict(title='가격 (USDT)', side='left', autorange=True, fixedrange=False), yaxis2=dict(title='거래량', overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified', margin=dict(l=50, r=50, t=50, b=50))
    fig.update_yaxes(title_text="가격 (USDT)", row=1, col=1, secondary_y=False); fig.update_yaxes(title_text="거래량", row=1, col=1, secondary_y=True, showgrid=False)
    for i in range(1, rows + 1): fig.update_xaxes(showticklabels=True, row=i, col=1)
    return fig

# --- 실시간 가격 조회 함수 ---
@st.cache_data(ttl=5) # 5초 캐시
def get_live_price(symbol):
    try:
        client = Client("","") # 키 없이 조회
        ticker = client.futures_ticker(symbol=symbol)
        return float(ticker.get('lastPrice'))
    except Exception:
        return None

# --- UI 컨트롤 ---
cols_top = st.columns([2, 1, 1, 1, 3]) # 컬럼 재구성
selected_symbol = cols_top[0].selectbox("심볼 선택", options=POPULAR_FUTURES_SYMBOLS, key="rt_symbol", index=POPULAR_FUTURES_SYMBOLS.index(st.session_state.get('rt_symbol', DEFAULT_SYMBOL)))
selected_interval_key = cols_top[1].selectbox("봉 주기 선택", options=list(INTERVAL_OPTIONS.keys()), key="rt_interval", index=list(INTERVAL_OPTIONS.keys()).index(st.session_state.get('rt_interval', DEFAULT_INTERVAL_KEY)))
selected_limit = cols_top[2].number_input("캔들 개수", min_value=50, max_value=1000, value=st.session_state.get('rt_limit', DEFAULT_LIMIT), step=10, key="rt_limit", help="표시할 캔들 개수")

# ★★★ 현재가 표시 Metric 추가 ★★★
current_live_price = get_live_price(selected_symbol)
price_display = f"{current_live_price:.4f}" if current_live_price else "N/A"
cols_top[3].metric("현재가", price_display) # 네 번째 컬럼에 현재가 표시

selected_indicators = cols_top[4].multiselect("지표 선택", options=INDICATOR_OPTIONS, default=DEFAULT_INDICATORS, key="rt_indicators") # 다섯 번째 컬럼으로 이동

interval_value = INTERVAL_OPTIONS[selected_interval_key]

# --- 자동 새로고침 ---
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="realtime_chart_refresh")

# --- 차트 표시 영역 ---
chart_placeholder = st.empty()

# --- 데이터 로드 및 차트 업데이트 ---
with st.spinner(f"{selected_symbol} ({selected_interval_key}) 데이터 로딩 및 지표 계산 중..."):
    df_klines, error_msg = fetch_and_prepare_data(selected_symbol, interval_value, selected_limit)

if error_msg: chart_placeholder.error(error_msg)
elif df_klines is not None and not df_klines.empty:
    try:
        fig = plot_candlestick_with_indicators(df_klines, selected_symbol, selected_interval_key, selected_indicators)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    except Exception as plot_error: chart_placeholder.error(f"차트 생성 중 오류: {plot_error}"); st.code(traceback.format_exc())
else: chart_placeholder.warning("차트 데이터를 표시할 수 없습니다.")

st.caption(f"최근 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (자동 새로고침 간격: {REFRESH_INTERVAL_MS/1000}초)")