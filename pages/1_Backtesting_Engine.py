# pages/1_Backtesting_Engine.py (거래로그 키 수정, 백테스트 최종가 표시 추가)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from binance.client import Client
import numpy as np
import itertools
import time
import traceback

# --- Initialize State ---
# ... (동일) ...
if 'last_backtest_result' not in st.session_state: st.session_state.last_backtest_result = None
if 'last_backtest_params' not in st.session_state: st.session_state.last_backtest_params = {}
if 'selected_strategy' not in st.session_state: st.session_state.selected_strategy = 'RSI'
if 'best_opt_result' not in st.session_state: st.session_state.best_opt_result = None
if 'best_opt_params' not in st.session_state: st.session_state.best_opt_params = {}
if 'optimization_running' not in st.session_state: st.session_state.optimization_running = False

# --- 심볼 목록 정의 ---
# ... (동일) ...
POPULAR_FUTURES_SYMBOLS = sorted(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT","AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "TRXUSDT", "LTCUSDT", "ETCUSDT","BCHUSDT", "NEARUSDT", "OPUSDT", "APTUSDT", "ARBUSDT", "FILUSDT", "ATOMUSDT","UNIUSDT", "XLMUSDT", "APEUSDT", "EOSUSDT", "AAVEUSDT", "FTMUSDT", "SANDUSDT","MANAUSDT", "AXSUSDT", "1000PEPEUSDT", "1000SHIBUSDT"])
DEFAULT_SYMBOL_BT = "XRPUSDT"; DEFAULT_INTERVAL_KEY_BT = "1h"

# --- Backtesting Utility: Calculate Metrics ---
def calculate_metrics(equity_series, closed_trades_details, initial_capital):
    # ... (동일) ...
    final_equity = equity_series.iloc[-1] if not equity_series.empty else initial_capital
    if initial_capital == 0: total_return_percent = 0.0
    else: total_return_percent = (final_equity - initial_capital) / initial_capital * 100
    if not equity_series.empty and initial_capital > 0:
        roll_max = equity_series.cummax(); roll_max[roll_max == 0] = 1e-10; daily_drawdown = equity_series / roll_max - 1.0
        max_drawdown = daily_drawdown.cummin().min()
        if not isinstance(max_drawdown, (int, float)) or np.isnan(max_drawdown): max_drawdown = 0.0
        max_drawdown = min(0.0, max_drawdown)
    else: max_drawdown = 0.0
    num_trades = len(closed_trades_details);
    if num_trades > 0:
        profitable_trades = sum(1 for t in closed_trades_details if t.get('PnL', 0) > 0); win_rate = (profitable_trades / num_trades) * 100
        gross_profit = sum(t.get('PnL', 0) for t in closed_trades_details if t.get('PnL', 0) > 0); gross_loss = abs(sum(t.get('PnL', 0) for t in closed_trades_details if t.get('PnL', 0) <= 0))
        if gross_loss > 0: profit_factor = gross_profit / gross_loss
        elif gross_profit > 0: profit_factor = float('inf')
        else: profit_factor = 0.0
    else: win_rate = 0.0; profit_factor = 0.0
    sharpe_ratio = 0.0; sortino_ratio = 0.0; trading_days_per_year = 252
    if not equity_series.empty and len(equity_series) > 1:
        daily_returns = equity_series.resample('D').last().pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1:
                 downside_std_dev = downside_returns.std()
                 if downside_std_dev != 0: sortino_ratio = (daily_returns.mean() / downside_std_dev) * np.sqrt(trading_days_per_year)
                 elif daily_returns.mean() > 0 : sortino_ratio = float('inf')
    return {'final_equity': final_equity, 'total_return_percent': total_return_percent, 'max_drawdown': max_drawdown, 'win_rate': win_rate, 'profit_factor': profit_factor, 'num_trades': num_trades, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio}


# --- Backtesting Function: RSI Strategy ---
def run_rsi_backtest(params):
    # ... (이전과 동일) ...
    try:
        client = Client("", ""); start_dt = datetime.utcnow() - timedelta(days=params['period_days']); klines = client.get_historical_klines(params['symbol'], params['interval'], start_dt.strftime("%d %b %Y %H:%M:%S"))
        if not klines: return None, f"{params['symbol']} ({params['interval']}) 데이터 로드 실패."
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms'); df.set_index('Open time', inplace=True)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']; df = df[ohlcv_cols].copy(); df = df.apply(pd.to_numeric, errors='coerce'); df.dropna(subset=ohlcv_cols, inplace=True)
        if df.empty: return None, "숫자 변환 후 유효 데이터 없음."
        df.ta.rsi(length=params['rsi_period'], append=True); df.ta.sma(length=20, append=True); df.ta.sma(length=60, append=True); df.dropna(inplace=True)
        if df.empty: return None, "지표 계산 후 데이터 부족."
        rsi_col_name = f'RSI_{params["rsi_period"]}'; sma_20_col = 'SMA_20'; sma_60_col = 'SMA_60'; required_cols = [rsi_col_name, sma_20_col, sma_60_col]
        if not all(col in df.columns for col in required_cols): return None, f"필수 지표 컬럼({required_cols}) 없음."
        cash = params['capital']; fee = params['fee']; slippage = params.get('slippage', 0.0) # 슬리피지 파라미터 가져오기
        position_size = 0.0; entry_price = 0.0; entry_time = None; equity = [cash]; buy_signals, sell_signals = [], []; closed_trades_details = []; position_type = "NONE"
        for i, row in df.iterrows():
            current_price = row['Close']; current_rsi = row[rsi_col_name]; current_time = i
            if position_type == "LONG" and current_rsi > params['rsi_sell']:
                exit_price_slippage = current_price * (1 - slippage) # 슬리피지 적용
                sell_value = exit_price_slippage * position_size; trade_fee = sell_value * fee; pnl = (exit_price_slippage - entry_price) * position_size - trade_fee
                cash += sell_value - trade_fee; closed_trades_details.append({'Entry Time': entry_time, 'Exit Time': current_time, 'Side': 'LONG', 'Entry Price': entry_price, 'Exit Price': exit_price_slippage, 'Size': position_size, 'PnL': pnl})
                sell_signals.append((current_time, exit_price_slippage)); position_size = 0.0; entry_price = 0.0; entry_time = None; position_type = "NONE"
            elif position_type == "NONE" and current_rsi < params['rsi_buy']:
                entry_price_slippage = current_price * (1 + slippage) # 슬리피지 적용
                investment_amount = cash; trade_fee = investment_amount * fee; actual_investment = investment_amount - fee
                if actual_investment > 0 and entry_price_slippage > 0:
                    position_size = actual_investment / entry_price_slippage; cash = 0.0; entry_price = entry_price_slippage; entry_time = current_time; position_type = "LONG"
                    buy_signals.append((current_time, entry_price_slippage))
            current_equity = cash + (position_size * current_price); equity.append(current_equity)
        equity_index_start = df.index[0] - (df.index[1] - df.index[0]) if len(df.index) > 1 else df.index[0]
        equity_series = pd.Series(equity, index=[equity_index_start] + list(df.index)); equity_series = equity_series[~equity_series.index.duplicated(keep='last')]
        metrics = calculate_metrics(equity_series, closed_trades_details, params['capital'])
        return {'df': df, 'equity_series': equity_series, 'trades': closed_trades_details, 'buy_signals': buy_signals, 'sell_signals': sell_signals, 'rsi_column': rsi_col_name, 'sma_20_col': sma_20_col, 'sma_60_col': sma_60_col, **metrics}, None
    except Exception as e: import traceback; error_details = traceback.format_exc(); return None, f"RSI 백테스팅 오류: {e}\n상세:\n{error_details}"

# --- Backtesting Function: Candle Pattern Strategy (슬리피지 적용) ---
def run_candle_pattern_backtest(params):
    try:
        # ... (데이터 로드, 캔들 색깔 정의 부분 동일) ...
        client = Client("", ""); interval = Client.KLINE_INTERVAL_1DAY; start_dt = datetime.utcnow() - timedelta(days=params['period_days']); klines = client.get_historical_klines(params['symbol'], interval, start_dt.strftime("%d %b %Y %H:%M:%S"))
        if not klines: return None, f"{params['symbol']} (1d) 데이터 로드 실패."
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms'); df.set_index('Open time', inplace=True)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']; df = df[ohlcv_cols].copy(); df = df.apply(pd.to_numeric, errors='coerce'); df.dropna(subset=ohlcv_cols, inplace=True)
        if df.empty: return None, "숫자 변환 후 유효 데이터 없음."
        df['Color'] = 0; df.loc[df['Close'] > df['Open'], 'Color'] = 1; df.loc[df['Close'] < df['Open'], 'Color'] = -1
        if len(df) < 3: return None, "백테스팅 위한 최소 데이터 부족 (3일 이상 필요)."

        cash = params['capital']; fee = params['fee']; slippage = params.get('slippage', 0.0) # 슬리피지
        position_type = "NONE"; position_size = 0.0; entry_price = 0.0; entry_time = None
        equity = [cash]; buy_signals, sell_signals = [], []; closed_trades_details = []

        for i in range(2, len(df)):
            current_price = df['Close'].iloc[i]; current_time = df.index[i]; prev_color1 = df['Color'].iloc[i-1]; prev_color2 = df['Color'].iloc[i-2]
            enter_signal = "NONE"; exit_signal = False; two_green = prev_color1 == 1 and prev_color2 == 1; two_red = prev_color1 == -1 and prev_color2 == -1
            if position_type == "NONE":
                if two_green: enter_signal = "SHORT"
                elif two_red: enter_signal = "LONG"
            elif position_type == "LONG":
                if two_green: exit_signal = True
            elif position_type == "SHORT":
                if two_red: exit_signal = True

            if exit_signal and position_size > 0:
                exit_side = 'SELL' if position_type == "LONG" else 'BUY'
                if exit_side == 'SELL': exit_price_slippage = current_price * (1 - slippage) # 슬리피지
                else: exit_price_slippage = current_price * (1 + slippage)
                proceeds = exit_price_slippage * position_size; trade_fee = proceeds * fee; profit = 0
                if position_type == "LONG": profit = (exit_price_slippage - entry_price) * position_size - trade_fee
                elif position_type == "SHORT": profit = (entry_price - exit_price_slippage) * position_size - trade_fee
                cash += proceeds - trade_fee
                closed_trades_details.append({'Entry Time': entry_time, 'Exit Time': current_time, 'Side': position_type, 'Entry Price': entry_price, 'Exit Price': exit_price_slippage, 'Size': position_size, 'PnL': profit})
                if exit_side == 'SELL': sell_signals.append((current_time, exit_price_slippage))
                else: buy_signals.append((current_time, exit_price_slippage))
                position_type = "NONE"; position_size = 0.0; entry_price = 0.0; entry_time = None
            elif enter_signal != "NONE" and position_type == "NONE":
                entry_side = 'BUY' if enter_signal == "LONG" else 'SELL'
                if entry_side == 'BUY': entry_price_slippage = current_price * (1 + slippage) # 슬리피지
                else: entry_price_slippage = current_price * (1 - slippage)
                investment_amount = cash ; trade_fee = investment_amount * fee; actual_investment = investment_amount - fee
                if actual_investment > 0 and entry_price_slippage > 0:
                    position_size = actual_investment / entry_price_slippage; cash = 0.0
                    entry_price = entry_price_slippage; position_type = enter_signal; entry_time = current_time
                    if entry_side == 'BUY': buy_signals.append((current_time, entry_price))
                    else: sell_signals.append((current_time, entry_price))
            current_equity = cash + (position_size * current_price); equity.append(current_equity)

        # equity_series 생성 (최종 수정 적용됨)
        equity_series = pd.Series(equity[1:], index=df.index[1:]) # 최종 수정된 인덱스 사용

        metrics = calculate_metrics(equity_series, closed_trades_details, params['capital']) # 상세 거래 내역 전달
        return {'df': df, 'equity_series': equity_series, 'trades': closed_trades_details, 'buy_signals': buy_signals, 'sell_signals': sell_signals, **metrics}, None
    except Exception as e: import traceback; error_details = traceback.format_exc(); return None, f"캔들 패턴 백테스팅 오류: {e}\n상세:\n{error_details}"


# --- 앱 실행 및 UI 구성 ---
st.header("⚙️ 백테스팅 엔진")
selected_strategy = st.selectbox("전략 선택", options=["RSI", "Consecutive Candles"], key="selected_strategy")

# --- 개별 백테스팅 Form (슬리피지 입력 추가) ---
with st.form("backtest_form"):
    st.subheader(f"{selected_strategy} 전략 - 개별 백테스트")
    col1, col2, col3, col4 = st.columns(4) # 컬럼 4개 사용
    selected_symbol_bt = col1.selectbox("심볼", options=POPULAR_FUTURES_SYMBOLS, key='backtest_symbol', index=POPULAR_FUTURES_SYMBOLS.index(st.session_state.get('backtest_symbol', DEFAULT_SYMBOL_BT)))
    interval_options = {"1m": Client.KLINE_INTERVAL_1MINUTE, "5m": Client.KLINE_INTERVAL_5MINUTE, "15m": Client.KLINE_INTERVAL_15MINUTE, "30m": Client.KLINE_INTERVAL_30MINUTE, "1h": Client.KLINE_INTERVAL_1HOUR, "4h": Client.KLINE_INTERVAL_4HOUR, "1d": Client.KLINE_INTERVAL_1DAY}
    interval_disabled = (selected_strategy == "Consecutive Candles")
    interval_key = col1.selectbox("봉 주기 (RSI 전략용)", options=list(interval_options.keys()), index=list(interval_options.keys()).index(st.session_state.get('backtest_interval_key', DEFAULT_INTERVAL_KEY_BT)), key='backtest_interval_key', help="연속 캔들 전략은 일봉(1d)으로 고정됩니다.", disabled=interval_disabled)
    interval = interval_options[interval_key] if not interval_disabled else Client.KLINE_INTERVAL_1DAY
    col1.number_input("기간 (일)", min_value=3, max_value=1000, key='backtest_period_days', value=st.session_state.get('backtest_period_days', 90))

    if selected_strategy == "RSI":
        col2.number_input("RSI 기간", min_value=2, key='rsi_period', value=st.session_state.get('rsi_period', 14))
        col2.number_input("RSI 매수 기준 (<)", key='rsi_buy_threshold', value=st.session_state.get('rsi_buy_threshold', 30.0), step=0.1, format="%.1f")
        col2.number_input("RSI 매도 기준 (>)", key='rsi_sell_threshold', value=st.session_state.get('rsi_sell_threshold', 50.0), step=0.1, format="%.1f")
    else: col2.info("캔들 패턴 전략 파라미터 없음")

    col3.number_input("초기 자본금 ($)", min_value=1.0, format="%.2f", key='initial_capital', value=st.session_state.get('initial_capital', 10000.0))
    fee_input = col3.number_input("거래 수수료 (%)", min_value=0.0, value=st.session_state.get('fee_percent', 0.1)*100, format="%.3f", key='fee_input')
    fee = fee_input / 100.0
    # ★★★ 슬리피지 입력 추가 ★★★
    slippage_input = col3.number_input("슬리피지 (%)", min_value=0.0, value=st.session_state.get('slippage_percent', 0.01)*100, format="%.3f", key='slippage_input', help="체결 시 예상되는 가격 미끄러짐 비율")
    slippage = slippage_input / 100.0 # 비율로 변환

    submitted_single = st.form_submit_button("▶️ 개별 백테스트 실행")

# --- 파라미터 최적화 Form ---
st.markdown("---"); st.subheader("RSI 전략 파라미터 최적화")
if selected_strategy != "RSI": st.warning("파라미터 최적화는 현재 RSI 전략에 대해서만 지원됩니다.")
else:
    with st.form("optimize_form"):
        # ... (UI 동일) ...
        st.write("테스트할 RSI 파라미터 범위를 설정하세요.")
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        p_start = opt_col1.number_input("RSI 기간 시작", min_value=5, value=7, key="opt_p_start"); p_end = opt_col1.number_input("RSI 기간 끝", min_value=p_start, value=21, key="opt_p_end"); p_step = opt_col1.number_input("RSI 기간 간격", min_value=1, value=7, key="opt_p_step")
        b_start = opt_col2.number_input("RSI 매수 기준 시작 (<)", value=20.0, step=5.0, format="%.1f", key="opt_b_start"); b_end = opt_col2.number_input("RSI 매수 기준 끝 (<)", value=40.0, step=5.0, format="%.1f", key="opt_b_end"); b_step = opt_col2.number_input("RSI 매수 기준 간격", value=5.0, step=1.0, format="%.1f", key="opt_b_step")
        s_start = opt_col3.number_input("RSI 매도 기준 시작 (>)", value=50.0, step=5.0, format="%.1f", key="opt_s_start"); s_end = opt_col3.number_input("RSI 매도 기준 끝 (>)", value=80.0, step=5.0, format="%.1f", key="opt_s_end"); s_step = opt_col3.number_input("RSI 매도 기준 간격", value=10.0, step=1.0, format="%.1f", key="opt_s_step")
        submitted_optimize = st.form_submit_button("🚀 최적 파라미터 찾기")

        if submitted_optimize:
            # ... (최적화 실행 로직 - slippage 파라미터 전달 추가) ...
            st.session_state.optimization_running = True; # ... 상태 초기화 ...
            st.markdown("---"); st.subheader("⚙️ RSI 파라미터 최적화 진행 중...")
            rsi_periods = list(range(p_start, p_end + 1, p_step)); buy_thresholds = [round(x, 1) for x in np.arange(b_start, b_end + b_step, b_step)]; sell_thresholds = [round(x, 1) for x in np.arange(s_start, s_end + s_step, s_step)]
            param_combinations = list(itertools.product(rsi_periods, buy_thresholds, sell_thresholds)); total_combinations = len(param_combinations)
            st.write(f"총 {total_combinations}개의 파라미터 조합을 테스트합니다...");
            if total_combinations > 0 and total_combinations < 1000:
                opt_progress_bar = st.progress(0, text="최적화 진행률"); best_result_so_far = None; best_params_so_far = None; best_return = -float('inf'); results_log = []
                for i, (p, b, s) in enumerate(param_combinations):
                    if b >= s: continue
                    # ★★★ 최적화 실행 시 slippage 전달 ★★★
                    current_params = {'symbol': st.session_state.backtest_symbol.upper(), 'interval': interval, 'period_days': st.session_state.backtest_period_days, 'capital': st.session_state.initial_capital, 'fee': fee, 'slippage': slippage, 'strategy': 'RSI_Optimization', 'rsi_period': p, 'rsi_buy': b, 'rsi_sell': s, 'interval_key': st.session_state.backtest_interval_key}
                    result, error = run_rsi_backtest(current_params)
                    if result and not error:
                        current_return = result['total_return_percent']; results_log.append({'period': p, 'buy': b, 'sell': s, 'return': current_return})
                        if current_return > best_return: best_return = current_return; best_result_so_far = result; best_params_so_far = current_params
                    progress_text = f"최적화 진행률 ({i+1}/{total_combinations})"; opt_progress_bar.progress((i + 1) / total_combinations, text=progress_text)
                opt_progress_bar.progress(100, text="최적화 완료!")
                if best_result_so_far: st.session_state.best_opt_result = best_result_so_far; st.session_state.best_opt_params = best_params_so_far; st.success("최적 파라미터 검색 완료!")
                else: st.warning("최적 파라미터를 찾지 못했습니다."); st.session_state.best_opt_result = None; st.session_state.best_opt_params = {}
            else: st.warning(f"테스트할 조합이 너무 많거나(>{1000}) 없습니다.")
            st.session_state.optimization_running = False
            if submitted_optimize: time.sleep(0.1); st.rerun()


# 결과 표시 컨테이너
results_container = st.container()

# --- 개별 백테스트 버튼 처리 ---
if submitted_single:
    st.session_state.optimization_running = False; st.session_state.best_opt_result = None; st.session_state.best_opt_params = {}
    with results_container:
         results_container.empty(); st.info("개별 백테스트 실행 중...")
         # ★★★ params 에 slippage 추가 ★★★
         current_params = {'symbol': selected_symbol_bt, 'interval': interval, 'period_days': st.session_state.backtest_period_days, 'capital': st.session_state.initial_capital, 'fee': fee, 'slippage': slippage, 'strategy': selected_strategy, 'interval_key': interval_key}
         if selected_strategy == "RSI": current_params.update({'rsi_period': st.session_state.rsi_period, 'rsi_buy': st.session_state.rsi_buy_threshold, 'rsi_sell': st.session_state.rsi_sell_threshold})
         if selected_strategy == "RSI": result, error = run_rsi_backtest(current_params)
         elif selected_strategy == "Consecutive Candles": result, error = run_candle_pattern_backtest(current_params)
         else: result, error = None, "알 수 없는 전략입니다."
         if error: st.error(f"백테스트 실패: {error}"); st.session_state.last_backtest_result = None; st.session_state.last_backtest_params = {}
         elif result: st.success(f"{selected_strategy} 전략 백테스트 완료!"); st.session_state.last_backtest_result = result; st.session_state.last_backtest_params = current_params


# --- 최종 결과 표시 로직 (★★★ 상세 거래 내역 표시 수정 ★★★) ---
result_to_display = None; params_to_display = None; title_prefix = ""
if st.session_state.get('best_opt_result'): result_to_display = st.session_state.best_opt_result; params_to_display = st.session_state.best_opt_params; title_prefix = "🏆 최적 파라미터 결과"
elif st.session_state.get('last_backtest_result'): result_to_display = st.session_state.last_backtest_result; params_to_display = st.session_state.last_backtest_params; title_prefix = "📊 개별 백테스트 결과"

if result_to_display and params_to_display:
    try:
        with results_container:
            run_strategy = params_to_display.get('strategy', 'Unknown'); st.markdown("---"); st.subheader(f"{title_prefix} ({run_strategy} Strategy)")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("최종 자산", f"${result_to_display['final_equity']:,.2f}"); m_col1.metric("총 수익률", f"{result_to_display['total_return_percent']:.2f}%")
            m_col2.metric("최대 낙폭", f"{result_to_display['max_drawdown']*100:.2f}%"); m_col2.metric("승률", f"{result_to_display['win_rate']:.2f}%")
            pf_display = f"{result_to_display['profit_factor']:.2f}";
            if result_to_display['profit_factor'] == float('inf'): pf_display = "∞"
            elif result_to_display['num_trades'] == 0 : pf_display = "N/A"
            elif result_to_display['profit_factor'] == 0 : pf_display = "0.00"
            m_col3.metric("손익비", pf_display); m_col3.metric("총 거래 수", result_to_display['num_trades'])
            m_col4.metric("샤프 지수", f"{result_to_display.get('sharpe_ratio', 0.0):.2f}")
            m_col4.metric("소르티노 지수", f"{result_to_display.get('sortino_ratio', 0.0):.2f}")
            # ★★★ 백테스트 최종 종가 표시 추가 ★★★
            last_close_price = result_to_display['df']['Close'].iloc[-1] if not result_to_display['df'].empty else 0
            m_col5.metric("최종 종가", f"{last_close_price:.4f}")
            # ★★★ 추가 끝 ★★★
            if title_prefix.startswith("🏆"): best_p = params_to_display.get('rsi_period', '?'); best_b = params_to_display.get('rsi_buy', '?'); best_s = params_to_display.get('rsi_sell', '?'); m_col5.markdown("**Best Params:**"); m_col5.markdown(f"- RSI Prd: **{best_p}**\n- Buy Thr: **{best_b}**\n- Sell Thr: **{best_s}**")

            st.subheader("📈 Charts")
            # ... (차트 UI 동일) ...
            show_rsi_chart = (run_strategy.startswith("RSI")); fig_rows = 3 if show_rsi_chart else 2; row_heights = [0.6, 0.4] if not show_rsi_chart else [0.5, 0.2, 0.3]; subplot_titles = [f"{params_to_display.get('symbol','')} 가격 & 신호"];
            if show_rsi_chart: subplot_titles.append(f"RSI ({params_to_display.get('rsi_period','')})")
            subplot_titles.append("자산 곡선")
            fig = make_subplots(rows=fig_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=subplot_titles, row_heights=row_heights, specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (fig_rows - 1))
            df_chart = result_to_display['df']
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='가격'), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='거래량', marker_color='rgba(150, 150, 150, 0.5)'), row=1, col=1, secondary_y=True)
            if run_strategy.startswith("RSI"):
                sma_20_col = result_to_display.get('sma_20_col', 'SMA_20'); sma_60_col = result_to_display.get('sma_60_col', 'SMA_60');
                if sma_20_col in df_chart.columns: fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[sma_20_col], name='SMA 20', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)), row=1, col=1, secondary_y=False)
                if sma_60_col in df_chart.columns: fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[sma_60_col], name='SMA 60', line=dict(color='rgba(135, 206, 250, 0.8)', width=1)), row=1, col=1, secondary_y=False)
            if result_to_display['buy_signals']: buy_times, buy_prices = zip(*result_to_display['buy_signals']); fig.add_trace(go.Scatter(x=list(buy_times), y=list(buy_prices), mode='markers', marker=dict(color='lime', size=10, symbol='triangle-up'), name='매수'), row=1, col=1, secondary_y=False)
            if result_to_display['sell_signals']: sell_times, sell_prices = zip(*result_to_display['sell_signals']); fig.add_trace(go.Scatter(x=list(sell_times), y=list(sell_prices), mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='매도'), row=1, col=1, secondary_y=False)
            if show_rsi_chart:
                rsi_row = 2; fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart[result_to_display['rsi_column']], name='RSI', line=dict(color='rgba(188, 128, 189, 0.8)')), row=rsi_row, col=1)
                fig.add_hline(y=params_to_display.get('rsi_buy', 30), line_dash="dash", line_color="rgba(0, 255, 0, 0.5)", annotation_text=f"매수 기준 ({params_to_display.get('rsi_buy', 30)})", annotation_position="bottom right", row=rsi_row, col=1)
                fig.add_hline(y=params_to_display.get('rsi_sell', 70), line_dash="dash", line_color="rgba(255, 0, 0, 0.5)", annotation_text=f"매도 기준 ({params_to_display.get('rsi_sell', 70)})", annotation_position="top right", row=rsi_row, col=1)
                fig.update_yaxes(title_text="RSI", row=rsi_row, col=1)
            equity_row = 3 if show_rsi_chart else 2; fig.add_trace(go.Scatter(x=result_to_display['equity_series'].index, y=result_to_display['equity_series'], name='자산', line=dict(color='rgba(0, 0, 255, 0.8)')), row=equity_row, col=1); fig.update_yaxes(title_text="자산 ($)", row=equity_row, col=1)
            chart_title = f"{params_to_display.get('symbol','')} {run_strategy} 백테스트 ({params_to_display.get('interval_key','?')}, {params_to_display.get('period_days','?')} 일)"; fig.update_layout(height=800, title_text=chart_title, xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)); fig.update_yaxes(title_text="가격 ($)", row=1, col=1, secondary_y=False); fig.update_yaxes(title_text="거래량", row=1, col=1, secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

            # ★★★ 상세 거래 내역 표시 (Key 수정) ★★★
            stable_key_suffix = f"{run_strategy}_{params_to_display.get('symbol','').lower()}" # 시간 대신 사용
            if st.checkbox("상세 거래 내역 보기", key=f"show_log_{stable_key_suffix}"): # 고정된 키 사용
                 if result_to_display['trades']:
                     trades_df = pd.DataFrame(result_to_display['trades'])
                     trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                     trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                     display_cols = ['Entry Time', 'Exit Time', 'Side', 'Entry Price', 'Exit Price', 'Size', 'PnL']
                     trades_df = trades_df[display_cols]; trades_df.rename(columns={'Entry Time':'진입 시간', 'Exit Time':'청산 시간', 'Side':'포지션', 'Entry Price':'진입 가격', 'Exit Price':'청산 가격', 'Size':'수량', 'PnL':'손익($)'}, inplace=True)
                     st.dataframe(trades_df.style.format({'진입 가격':'{:.4f}', '청산 가격':'{:.4f}', '수량':'{:.6f}', '손익($)':'{:.2f}'}), use_container_width=True)
                 else: st.info("실행된 거래가 없습니다.")

            if st.button("백테스트 결과 지우기"): st.session_state.last_backtest_result = None; st.session_state.best_opt_result = None; st.session_state.last_backtest_params = {}; st.session_state.best_opt_params = {}; st.rerun()
    except Exception as display_error:
         results_container.error(f"결과를 표시하는 중 오류가 발생했습니다: {display_error}")
         results_container.code(traceback.format_exc())

elif not st.session_state.get('optimization_running'):
     results_container.info("백테스트를 실행하거나 최적화를 진행해주세요.")