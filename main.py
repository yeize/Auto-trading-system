# main.py (파일 로깅, 포지션 크기 조절, TP/SL 적용 최종)

import configparser
import os
import time
import json
import logging # **** 파일 로깅 모듈 ****
from datetime import datetime
from binance_utils import (
    connect_binance, get_futures_balance, get_current_price,
    place_futures_order, get_position_info,
    get_symbol_trading_rules, format_quantity
)
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np # 계산에 필요할 수 있음
import traceback # 오류 로깅용

# --- 설정 ---
STATUS_FILE = 'status.json'
CONFIG_FILE = 'config.ini'
CHECK_INTERVAL_SECONDS = 3
BALANCE_CHECK_INTERVAL_SECONDS = 60 # 잔고 확인 주기
MAX_LOG_LINES = 20

# --- 로깅 설정 ---
def setup_logging(log_file='trading_bot.log', log_level='INFO'):
    level = getattr(logging, log_level.upper(), logging.INFO)
    # 루트 로거에 핸들러 추가 (중복 방지)
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # 파일 핸들러
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logging.info("="*50); logging.info("로깅 시스템 시작"); logging.info("="*50)
    else:
         logging.getLogger().setLevel(level)
         # logging.info("로깅 시스템 이미 설정됨.") # 시작 시 중복 로그 방지


# --- 유틸리티 함수 ---
def load_config(config_file=CONFIG_FILE):
    # ... (이전과 동일, 단 required_sections 업데이트) ...
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, config_file)
    print(f"설정 파일을 읽는 중: {abs_config_path}", flush=True) # 로거 설정 전 print 사용
    if not os.path.exists(abs_config_path): raise FileNotFoundError(f"설정 파일({config_file}) 없음")
    config = configparser.ConfigParser(); config.read(abs_config_path, encoding='utf-8')
    required_sections = { # risk_per_trade_percent 추가, order_usdt 제거
        'binance': ['api_key', 'api_secret'],
        'trading': ['symbol', 'entry_threshold_percent', 'take_profit_percent', 'stop_loss_percent', 'risk_per_trade_percent'],
        'logging': ['log_file', 'log_level']
        }
    for section, keys in required_sections.items():
         if section not in config: raise ValueError(f"설정 파일에 [{section}] 섹션 없음")
         for key in keys:
              if key not in config[section]: raise ValueError(f"[{section}] 섹션에 '{key}' 키 없음")
    return config

def save_status(data):
    # ... (이전과 동일) ...
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e: logging.error(f"오류: 상태 파일 저장 실패 - {e}", exc_info=True)


def load_status():
    # ... (이전과 동일) ...
    default_status = {"current_position": "NONE", "entry_price": None, "position_amount": 0.0, "base_price": None, "available_balance": 0.0, "log_messages": []}
    if not os.path.exists(STATUS_FILE): return default_status
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f: content = f.read()
        if not content: return default_status
        loaded_data = json.loads(content)
        for key, value in default_status.items():
             if key not in loaded_data: loaded_data[key] = value
        loaded_data["log_messages"] = loaded_data.get("log_messages", [])[-MAX_LOG_LINES:]
        return loaded_data
    except Exception as e: logging.error(f"오류: 상태 파일 로드 실패 - {e}. 기본값 사용.", exc_info=True); return default_status


def log_and_update_status(level, message, status_data):
    """지정된 레벨로 로깅하고 status_data 로그 목록 업데이트."""
    log_func = getattr(logging, level.lower(), logging.debug)
    log_func(message) # 파일/콘솔 로깅
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    if "log_messages" not in status_data: status_data["log_messages"] = []
    status_data["log_messages"].append(log_entry)
    status_data["log_messages"] = status_data["log_messages"][-MAX_LOG_LINES:]


# --- ★★★ 포지션 크기 계산 함수 ★★★ ---
def calculate_position_size(balance, risk_percent, current_price, sl_percent, symbol_rules):
    """손절매 기준 리스크 기반 포지션 크기 계산"""
    if balance <= 0 or current_price <= 0 or sl_percent <= 0:
        logging.warning("포지션 크기 계산 불가: 잔고/가격/손절률이 0 이하입니다.")
        return 0.0

    risk_amount_usdt = balance * (risk_percent / 100.0)
    stop_loss_price_long = current_price * (1 - sl_percent / 100.0)
    stop_loss_price_short = current_price * (1 + sl_percent / 100.0)

    # 롱/숏 동일하게 거리 계산 (절대값)
    distance_to_sl_usd = abs(current_price - stop_loss_price_long) # 또는 short 사용해도 동일
    if distance_to_sl_usd == 0:
         logging.warning("포지션 크기 계산 불가: 손절 거리가 0입니다 (손절률 너무 작거나 가격 0).")
         return 0.0

    # 최대 가능 수량 (레버리지 미고려)
    position_size_crypto = risk_amount_usdt / distance_to_sl_usd
    logging.info(f"[Sizing] 잔고:{balance:.2f}, 리스크%:{risk_percent:.2f}, 손실허용:{risk_amount_usdt:.2f} USDT")
    logging.info(f"[Sizing] 현재가:{current_price:.4f}, SL가격(Long):{stop_loss_price_long:.4f}, SL거리:{distance_to_sl_usd:.4f}")
    logging.info(f"[Sizing] 계산된 수량 (미포맷): {position_size_crypto}")

    # 거래 규칙 적용하여 최종 수량 결정
    step_size = symbol_rules.get('stepSize')
    min_qty = float(symbol_rules.get('minQty', 0))
    min_notional = float(symbol_rules.get('minNotional', 0))

    formatted_qty = format_quantity(position_size_crypto, step_size)
    logging.info(f"[Sizing] 포맷팅된 수량: {formatted_qty} (StepSize: {step_size})")

    # 최소 주문 수량 및 최소 주문 금액 확인
    if formatted_qty < min_qty:
        logging.warning(f"계산된 수량({formatted_qty})이 최소 주문 수량({min_qty}) 미만입니다.")
        return 0.0
    if current_price * formatted_qty < min_notional:
        logging.warning(f"계산된 주문 금액({current_price * formatted_qty:.4f})이 최소 주문 금액({min_notional}) 미만입니다.")
        return 0.0

    logging.info(f"[Sizing] 최종 결정 수량: {formatted_qty}")
    return formatted_qty


# --- 메인 로직 ---
if __name__ == "__main__":
    # 로깅 설정 (try-except 추가)
    try:
        temp_config = load_config()
        log_file = temp_config['logging']['log_file']
        log_level = temp_config['logging']['log_level']
        setup_logging(log_file, log_level)
    except Exception as e:
        print(f"!!! 로깅 설정 오류: {e}. 기본 설정으로 계속합니다.", flush=True)
        setup_logging() # 기본값으로 설정

    logging.info("--- 트레이딩 봇 시작 (리스크 관리 적용) ---")
    client = None
    trading_rules = None
    base_price = None
    last_balance_check_time = 0

    status_data = {"bot_status": "INITIALIZING", "log_messages": []}

    try:
        # 1. 설정 로드
        config = load_config()
        api_key = config['binance']['api_key']; api_secret = config['binance']['api_secret']
        symbol = config['trading']['symbol'].upper()
        entry_threshold = float(config['trading']['entry_threshold_percent']) / 100.0
        take_profit_threshold = float(config['trading']['take_profit_percent']) / 100.0
        stop_loss_threshold = float(config['trading']['stop_loss_percent']) / 100.0
        risk_per_trade_percent = float(config['trading']['risk_per_trade_percent']) # 리스크 % 로드
        status_data['symbol'] = symbol

        log_and_update_status('info', f"설정 로드 완료: Symbol={symbol}, Risk={risk_per_trade_percent:.2f}%, Entry={entry_threshold*100:.2f}%, TP={take_profit_threshold*100:.2f}%, SL={stop_loss_threshold*100:.2f}%", status_data)

        # 2. 바이낸스 연결
        USE_TESTNET = False
        log_and_update_status('info', f"바이낸스 {'Testnet' if USE_TESTNET else 'Real Server'} 연결 시도...", status_data)
        client = connect_binance(api_key, api_secret, testnet=USE_TESTNET)

        if client:
            log_and_update_status('info', f"바이낸스 {'Testnet' if USE_TESTNET else 'Real Server'} 연결 성공.", status_data)
            status_data['bot_status'] = "CONNECTED"
            trading_rules = get_symbol_trading_rules(client, symbol)
            if not trading_rules: raise ValueError(f"{symbol} 거래 규칙 로드 실패.")
            log_and_update_status('info', f"{symbol} 거래 규칙 로드 완료.", status_data); save_status(status_data)

            # 3. 상태 로드
            loaded_status = load_status(); status_data.update(loaded_status)
            current_position = status_data.get("current_position", "NONE"); entry_price = status_data.get("entry_price")
            position_amount = status_data.get("position_amount", 0.0); base_price = status_data.get("base_price")
            available_balance = status_data.get("available_balance", 0.0)
            if current_position != "NONE" and position_amount <= 0:
                log_and_update_status('warning', f"로드 상태 불일치! Pos={current_position}, Amt={position_amount}. NONE 초기화.", status_data)
                current_position = "NONE"; entry_price = None; position_amount = 0.0; base_price = None
                status_data.update({"current_position": "NONE", "entry_price": None, "position_amount": 0.0, "base_price": None, "last_signal_display":"STATE_RESET"})
            log_and_update_status('info', f"로드 상태: Pos={current_position}, Entry={entry_price}, Amt={position_amount}, Base={base_price}, Bal={available_balance:.2f}", status_data); save_status(status_data)

            # 4. 메인 루프 시작
            log_and_update_status('info', "메인 루프 시작...", status_data)
            status_data['bot_status'] = "RUNNING"; save_status(status_data)
            last_balance_check_time = time.time()

            while True:
                save_needed = False; order_executed_this_loop = False
                current_timestamp = time.time()

                try:
                    current_price = get_current_price(client, symbol)
                    if current_price is None: log_and_update_status('warning', "현재 가격 조회 실패.", status_data); save_needed = True; time.sleep(CHECK_INTERVAL_SECONDS); continue
                    status_data['current_price'] = current_price

                    if current_position == "NONE" and base_price is None:
                        base_price = current_price; log_and_update_status('info', f"기준 가격 초기화: {base_price:.4f}", status_data); status_data['base_price'] = base_price; save_needed = True

                    # 잔고 확인 (주기적)
                    if current_timestamp - last_balance_check_time >= BALANCE_CHECK_INTERVAL_SECONDS:
                        balance_info = get_futures_balance(client, asset='USDT')
                        if balance_info and isinstance(balance_info.get('availableBalance'), str):
                            new_balance = float(balance_info['availableBalance'])
                            if abs(status_data.get('available_balance', 0.0) - new_balance) > 0.01:
                                 log_and_update_status('info', f"잔고 업데이트: {new_balance:.2f} USDT", status_data); status_data['available_balance'] = new_balance; save_needed = True
                            else: status_data['available_balance'] = new_balance # 값은 반영
                        else: log_and_update_status('warning', "잔고 조회 실패 또는 데이터 없음.", status_data); save_needed = True
                        last_balance_check_time = current_timestamp

                    # 상태 로깅 (주기적 DEBUG 레벨)
                    available_balance = status_data.get('available_balance', 0.0) # 최신 잔고 사용
                    base_price_display = f"{base_price:.4f}" if base_price is not None else "N/A"
                    status_message = f"가격:{current_price:.4f} | 기준:{base_price_display} | Pos:{current_position}({position_amount}) | 잔고:{available_balance:.2f}"
                    if entry_price and current_position != "NONE":
                         pnl_percent = 0.0; tp_price = None; sl_price = None
                         if entry_price != 0:
                             if current_position == "LONG": pnl_percent = ((current_price - entry_price) / entry_price) * 100; tp_price = entry_price * (1 + take_profit_threshold); sl_price = entry_price * (1 - stop_loss_threshold)
                             elif current_position == "SHORT": pnl_percent = ((entry_price - current_price) / entry_price) * 100; tp_price = entry_price * (1 - take_profit_threshold); sl_price = entry_price * (1 + stop_loss_threshold)
                         status_message += f" | Entry:{entry_price:.4f} | PnL:{pnl_percent:+.2f}%"
                         if tp_price: status_message += f" | TP:{tp_price:.4f}"
                         if sl_price: status_message += f" | SL:{sl_price:.4f}"
                    if not status_data["log_messages"] or not status_data["log_messages"][-1].endswith(status_message):
                         log_and_update_status('debug', status_message, status_data); save_needed = True

                    # --- 매매 전략 로직 (리스크 관리 추가) ---
                    order_result = None
                    if current_position == "NONE":
                        if base_price is not None:
                            price_change_percent_from_base = ((current_price - base_price) / base_price) * 100 if base_price != 0 else 0
                            enter_side = None
                            if price_change_percent_from_base >= entry_threshold * 100: enter_side = 'BUY' # LONG
                            elif price_change_percent_from_base <= -entry_threshold * 100: enter_side = 'SELL' # SHORT

                            if enter_side:
                                log_and_update_status('info', f"*** {enter_side} 진입 조건 ({price_change_percent_from_base:+.3f}%) ***", status_data); save_needed = True
                                # ★★★ 포지션 크기 계산 ★★★
                                formatted_qty = calculate_position_size(available_balance, risk_per_trade_percent, current_price, stop_loss_threshold, trading_rules)
                                # ★★★ 계산 끝 ★★★
                                if formatted_qty > 0: # 계산된 수량이 유효할 때만 주문 시도
                                    order_result = place_futures_order(client, symbol, enter_side, 'MARKET', quantity=formatted_qty)
                                    if order_result and not order_result.get('error'):
                                        pos_type = "LONG" if enter_side == "BUY" else "SHORT"
                                        # 실제 체결가/수량 반영 (API 응답 사용 고려)
                                        filled_qty = float(order_result.get('executedQty', formatted_qty)) # 실제 체결량 우선 사용
                                        avg_price = float(order_result.get('avgPrice', current_price)) # 평균 체결가 우선 사용
                                        log_and_update_status('warning', f"--- {pos_type} 진입 성공 (ID:{order_result.get('orderId')}, Qty:{filled_qty}, AvgPx:{avg_price:.4f}) ---", status_data)
                                        current_position = pos_type; entry_price = avg_price; position_amount = filled_qty; base_price = None # 진입 후 base 초기화
                                        status_data.update({"current_position": current_position, "entry_price": entry_price, "position_amount": position_amount, "base_price": base_price, "last_signal_display": f"ENTER_{pos_type}"})
                                        order_executed_this_loop = True
                                    else: log_and_update_status('error', f"!!! {enter_side} 진입 주문 실패: {order_result.get('message', 'Unknown')}", status_data)
                                else: log_and_update_status('warning', f"!!! {enter_side} 진입 주문 실행 안함 (계산된 수량: {formatted_qty})", status_data)
                                save_needed = True

                    elif current_position != "NONE" and entry_price is not None and position_amount > 0: # 포지션 있을 때 청산 조건 (TP/SL)
                        pnl_percent = 0.0; should_exit = False; exit_reason = ""; exit_side = ""
                        if entry_price != 0:
                             if current_position == "LONG":
                                 pnl_percent = ((current_price - entry_price) / entry_price) * 100; tp_price = entry_price * (1 + take_profit_threshold); sl_price = entry_price * (1 - stop_loss_threshold)
                                 if current_price >= tp_price: should_exit = True; exit_reason = f"TP({pnl_percent:+.2f}%)"; exit_side = 'SELL'
                                 elif current_price <= sl_price: should_exit = True; exit_reason = f"SL({pnl_percent:+.2f}%)"; exit_side = 'SELL'
                             elif current_position == "SHORT":
                                 pnl_percent = ((entry_price - current_price) / entry_price) * 100; tp_price = entry_price * (1 - take_profit_threshold); sl_price = entry_price * (1 + stop_loss_threshold)
                                 if current_price <= tp_price: should_exit = True; exit_reason = f"TP({pnl_percent:+.2f}%)"; exit_side = 'BUY'
                                 elif current_price >= sl_price: should_exit = True; exit_reason = f"SL({pnl_percent:+.2f}%)"; exit_side = 'BUY'

                        if should_exit:
                            log_and_update_status('info', f"*** {current_position} 포지션 종료 조건 ({exit_reason}) ***", status_data); save_needed = True
                            formatted_qty = format_quantity(position_amount, trading_rules.get('stepSize'))
                            if formatted_qty > 0 and formatted_qty >= float(trading_rules.get('minQty', 0)):
                                order_result = place_futures_order(client, symbol, exit_side, 'MARKET', quantity=formatted_qty, reduce_only=True)
                                if order_result and not order_result.get('error'):
                                    exit_position_type = current_position; log_and_update_status('warning', f"--- {exit_position_type} 종료 성공 (ID: {order_result.get('orderId')}, 이유: {exit_reason}) ---", status_data)
                                    current_position = "NONE"; entry_price = None; position_amount = 0.0; base_price = None # 종료 후 base 초기화
                                    status_data.update({"current_position": "NONE", "entry_price": None, "position_amount": 0.0, "base_price": base_price, "last_signal_display": f"EXIT_{exit_position_type}"})
                                    order_executed_this_loop = True
                                else: log_and_update_status('error', f"!!! {current_position} 종료 주문 실패: {order_result.get('message', 'Unknown')}", status_data)
                            else: log_and_update_status('warning', f"!!! {current_position} 종료 최소 주문 요건 미달 (수량:{formatted_qty})", status_data)
                            save_needed = True

                    # --- 상태 저장 ---
                    status_data['timestamp'] = current_timestamp
                    if save_needed or order_executed_this_loop:
                         status_data['bot_status'] = "RUNNING"
                         # 저장 전 최신 상태 다시 한번 update 호출로 정리
                         status_data.update({ "current_position": current_position, "entry_price": entry_price, "position_amount": position_amount, "base_price": base_price, "available_balance": status_data.get('available_balance') })
                         save_status(status_data)

                except Exception as e:
                    error_message = f"!!! 메인 루프 오류: {e}"
                    log_and_update_status('error', error_message, status_data)
                    status_data['bot_status'] = "ERROR"; save_status(status_data)
                    logging.exception("메인 루프 상세 오류:") # 파일 로그에 traceback 기록
                    time.sleep(CHECK_INTERVAL_SECONDS * 2)

                time.sleep(CHECK_INTERVAL_SECONDS)

        else: # 클라이언트 연결 실패
            log_and_update_status('critical', "!!! 바이낸스 연결 실패 !!!", status_data)
            status_data['bot_status'] = "CONNECTION FAILED"; save_status(status_data)

    except FileNotFoundError as e: logging.critical(f"설정 파일 없음: {e}")
    except ValueError as e: logging.critical(f"설정 또는 규칙 오류: {e}", exc_info=True)
    except KeyboardInterrupt: logging.info("\n--- 프로그램 종료 요청됨 ---")
    except Exception as e: logging.critical(f"!!! 치명적인 오류 발생: {e}", exc_info=True)
    finally:
         logging.info("봇 상태 최종 저장 시도...")
         final_status = {"bot_status": "STOPPED"}
         try: final_status.update(status_data)
         except NameError: pass # status_data 정의 전 오류 발생 시 무시
         except Exception as e: logging.error(f"최종 상태 업데이트 오류: {e}")
         save_status(final_status)
         logging.info("--- 봇 종료 완료 ---")