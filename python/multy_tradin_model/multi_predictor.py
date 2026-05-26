import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import tensorflow as tf
import time

# --- НАСТРОЙКИ ---
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY"]
TIMEFRAME_MAPPING = {
    "EURUSD": mt5.TIMEFRAME_M30,
    "GBPUSD": mt5.TIMEFRAME_M30,
    "USDCHF": mt5.TIMEFRAME_M30,
    "USDJPY": mt5.TIMEFRAME_M30,
}
SEQ_LEN = 96 
NUM_CANDLES_TO_DOWNLOAD = 200

# Эти файлы должны лежать в той же папке после обучения выше
MODEL_FILE = "model_multi_v1.h5"
SCALER_FEATURES_FILE = "scaler_features_multi.pkl"
SCALER_PRICE_FILE = "scaler_price_multi.pkl"
# ------------------

def get_current_signal_for_pair(symbol):
    """Функция прогноза для выбранной пары."""
    tf_mt5 = TIMEFRAME_MAPPING[symbol]
    
    if not mt5.initialize():
        return None, f"❌ Ошибка подключения к терминалу для {symbol}."
        
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, NUM_CANDLES_TO_DOWNLOAD)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None, f"❌ Нет данных от терминала для {symbol}."
        
    df = pd.DataFrame(rates)
    
    # --- РАСЧЕТ ИНДИКАТОРОВ (как обычно) ---
    close_prices = df['close']
    
    df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()
    
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    EMA_12 = close_prices.ewm(span=12, adjust=False).mean()
    EMA_26 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA_12 - EMA_26 
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df.dropna(inplace=True)
    
    if len(df) < SEQ_LEN:
        return None, f"❌ Недостаточно данных для {symbol}."
        
    features_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal']
    
    try:
        scaler_X = joblib.load(SCALER_FEATURES_FILE)
        scaler_y = joblib.load(SCALER_PRICE_FILE)
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        return None, f"❌ Ошибка загрузки модели: {e}"
        
    X = df[features_cols].values 
    data_scaled = scaler_X.transform(X)
    
    input_data = data_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features_cols))
    
    prediction_scaled = model.predict(input_data)
    
    predicted_price_normalized = prediction_scaled.flatten()[0]
    predicted_price_real = scaler_y.inverse_transform(np.array([[predicted_price_normalized]]))[0][0]
    
    current_price_real = df['close'].iloc[-1]
    
    price_diff_pips = int(round((predicted_price_real - current_price_real) * 100000)) # Для 4-5 знаков

    if price_diff_pips > 5: # Порог в пунктах (можно настроить)
        signal_text = "BUY"
        signal_color_code = "GREEN"
        pips_sign = "+"
    elif price_diff_pips < -5:
        signal_text = "SELL"
        signal_color_code = "RED"
        pips_sign = ""
    else:
        signal_text = "HOLD"
        signal_color_code = "YELLOW"
        pips_sign = ""
        
        
    return {
        'symbol': symbol,
        'current_price': current_price_real,
        'predicted_price': predicted_price_real,
        'pips_diff': pips_diff,
        'signal': signal_text,
        'color': signal_color_code,
        'pips_sign': pips_sign,
        'pips_value': abs(price_diff_pips)
    }, None

def main():
    print("Доступные валютные пары:", ", ".join(SYMBOLS))
    
    while True:
        user_symbol = input("\nВведите валютную пару (или 'exit' для выхода): ").upper()
        
        if user_symbol == 'EXIT':
            break
            
        if user_symbol not in SYMBOLS:
            print("❌ Пара не поддерживается. Выберите из списка.")
            continue

        result, error_msg = get_current_signal_for_pair(user_symbol)
        
        if error_msg:
            print(error_msg)
            continue

        # Красивый вывод в консоль с цветами ANSI
        color_start = {
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "YELLOW": "\033[93m"
        }[result['color']]
        
        color_end = "\033[0m"
        
        print(f"\n{'='*40}")
        print(f"📊 ПРОГНОЗ ДЛЯ ПАРЫ: {result['symbol']}")
        print('='*40)
        
        print(f"🕒 Текущая цена: {result['current_price']:.5f}")
        print(f"🔮 Прогноз через 1 свечу: {result['predicted_price']:.5f}")
        
        pips_str = f"{result['pips_sign']}{result['pips_value']} п."
        print(f"Разница в пунктах: {pips_str}")
        
        print(f"{color_start}🚩 СИГНАЛ: {result['signal']}{color_end}")
        
if __name__ == "__main__":
    main()