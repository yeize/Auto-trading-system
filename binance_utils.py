# binance_utils.py (디버깅 로그 추가 버전)

import math
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def connect_binance(api_key, api_secret, testnet=False):
    """바이낸스 클라이언트 생성 및 연결 테스트."""
    try:
        client = Client(api_key, api_secret, testnet=testnet)
        client.get_server_time()
        server_type = "테스트넷" if testnet else "실 서버"
        print(f"바이낸스 API 연결 성공! ({server_type})")
        return client
    except BinanceAPIException as e: print(f"바이낸스 API 오류: {e}"); return None
    except BinanceRequestException as e: print(f"바이낸스 요청 오류: {e}"); return None
    except Exception as e: print(f"알 수 없는 오류 발생: {e}"); return None

def get_futures_balance(client, asset='USDT'):
    """선물 계정 잔고 조회 (디버깅 로그 추가)."""
    if not client: return None
    try:
        # --- 디버깅 로그 추가 ---
        print("[DEBUG] Fetching futures account balance...")
        futures_balances = client.futures_account_balance()
        print(f"[DEBUG] Raw futures balances: {futures_balances}") # <-- API 결과 직접 출력
        # --- 디버깅 로그 끝 ---

        asset_balance = next((item for item in futures_balances if item['asset'] == asset), None)
        # --- 디버깅 로그 추가 ---
        print(f"[DEBUG] Filtered balance info for {asset}: {asset_balance}") # <-- 필터링된 결과 출력
        # --- 디버깅 로그 끝 ---
        return asset_balance # 성공 시 해당 자산 정보 반환, 없으면 None 반환
    except BinanceAPIException as e: print(f"선물 잔고 조회 API 오류: {e}"); return None
    except Exception as e: print(f"선물 잔고 조회 중 오류 발생: {e}"); return None

def get_current_price(client, symbol):
    """선물 시장 현재가 조회."""
    if not client: return None
    try:
        ticker = client.futures_ticker(symbol=symbol)
        price = ticker.get('lastPrice')
        return float(price) if price else None
    except Exception: return None # 오류 시 간단히 None 반환

def get_latest_rsi(client, symbol, interval=Client.KLINE_INTERVAL_1HOUR, rsi_period=14):
    """최근 완료된 봉의 RSI 값 계산."""
    if not client: return None
    try:
        limit = rsi_period + 50
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) < rsi_period: return None
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Close'] = pd.to_numeric(df['Close'])
        rsi = ta.rsi(df['Close'], length=rsi_period)
        if rsi is None or rsi.empty: return None
        return rsi.iloc[-2] if len(rsi) >= 2 else rsi.iloc[-1] # 마지막 완료 봉 기준
    except Exception as e: print(f"RSI 조회 오류 ({symbol}): {e}"); return None

def format_quantity(quantity, step_size):
    """수량을 거래 규칙에 맞게 포맷."""
    try:
        if step_size is None or float(step_size) <= 0: precision = 8
        else: precision = int(round(-math.log10(float(step_size))))
        factor = 10 ** precision
        return math.floor(quantity * factor) / factor
    except Exception as e: print(f"수량 포매팅 오류: {e}"); return 0 # 오류 시 0 반환

def place_futures_order(client, symbol, side, order_type, quantity, price=None, timeInForce=None, reduce_only=False):
    """선물 주문 제출."""
    if not client: return {'error': True, 'message': '클라이언트 연결 안됨'}
    try:
        params = {'symbol': symbol, 'side': side, 'type': order_type, 'quantity': quantity}
        if reduce_only: params['reduceOnly'] = 'true'
        if order_type == 'LIMIT':
            if price is None: raise ValueError("LIMIT 주문에는 가격 필요")
            params['price'] = price; params['timeInForce'] = timeInForce if timeInForce else 'GTC'
        print(f"--- 주문 시도 ---\n  {params}")
        order = client.futures_create_order(**params)
        print(f"--- 주문 성공 ---\n{order}")
        return order
    except BinanceAPIException as e: print(f"!!! 주문 API 오류 ({symbol}): {e}"); return {'error': True, 'message': str(e)}
    except ValueError as e: print(f"!!! 주문 값 오류 ({symbol}): {e}"); return {'error': True, 'message': str(e)}
    except Exception as e: print(f"!!! 주문 중 오류 발생 ({symbol}): {e}"); return {'error': True, 'message': str(e)}

def get_position_info(client, symbol):
    """선물 포지션 정보 조회."""
    if not client: return None
    try:
        positions = client.futures_position_information(symbol=symbol)
        return positions[0] if positions else None
    except Exception as e: print(f"포지션 정보 조회 오류: {e}"); return None

def get_symbol_trading_rules(client, symbol):
    """심볼 거래 규칙 조회."""
    if not client: return None
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                rules = {'symbol': s['symbol'], 'status': s['status'], 'quantityPrecision': s['quantityPrecision'], 'pricePrecision': s['pricePrecision'], 'filters': s['filters']}
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE': rules['stepSize'] = f['stepSize']; rules['minQty'] = f['minQty']
                    elif f['filterType'] == 'PRICE_FILTER': rules['tickSize'] = f['tickSize']; rules['minPrice'] = f['minPrice']
                    elif f['filterType'] == 'MIN_NOTIONAL': rules['minNotional'] = f.get('notional') # 선물 API 기준
                return rules
        return None
    except Exception as e: print(f"거래 규칙 조회 오류: {e}"); return None