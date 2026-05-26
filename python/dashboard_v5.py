import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import os
import time
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# --- КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ ---
st.set_page_config(
    page_title="📈 Forex Analyzer v5",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (ПУТИ) ---
OUTPUT_FOLDER = 'data'
MODEL_FILE = "model_v5.1.h5"
SCALER_FEATURES_FILE = "scaler_features_v5.1.pkl"
SCALER_PRICE_FILE = "scaler_price_v5.1.pkl"
HISTORY_FILE = "prediction_history.txt"

# --- ФУНКЦИИ (Логика из твоих скриптов) ---

def load_history_data():
    """Загружает историю сигналов для графика."""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE, names=['timestamp', 'price', 'prediction', 'signal'], header=None)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    return pd.DataFrame()

def get_current_signal():
    """
    Основная функция, которая имитирует работу твоего пайплайна:
    1. Скачивает данные.
    2. Считает индикаторы.
    3. Готовит данные для модели.
    4. Прогнозирует и возвращает сигнал.
    """
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15
    NUM_CANDLES = 192 # Берем 2 дня для анализа

    # 1. ПОДКЛЮЧЕНИЕ И СКАЧИВАНИЕ ДАННЫХ
    if not mt5.initialize():
        return None, "❌ Ошибка подключения к MT5."

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None, "❌ Нет данных от терминала."

    # 2. СОЗДАНИЕ DATAFRAME И РАСЧЕТ ИНДИКАТОРОВ (как в data_processor)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    close_prices = df['close']
    
    # Индикаторы
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

    df['Histogram'] = df['MACD'] - df['Signal']
    
    # Удаляем строки с NaN (где индикаторы еще не посчитались)
    df.dropna(inplace=True)
    
    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ (как в predictor)
    features_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Histogram']
    
    # Берем последние 193 строк (как в data_processor_v5.py) 
    df_processed = df[features_cols].tail(193)
    
    # Берем последние 96 свечей для модели (SEQ_LEN)
    seq_len = 96 
    if len(df_processed) < seq_len:
        return None, "❌ Недостаточно данных для анализа."
        
    X = df_processed.values

    try:
        scaler_X = joblib.load(SCALER_FEATURES_FILE)
        scaler_y = joblib.load(SCALER_PRICE_FILE)
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    except Exception as e:
        return None, f"❌ Ошибка загрузки модели или файло: {e}"

    data_scaled = scaler_X.transform(X)
    
    input_data = data_scaled[-seq_len:].reshape(1, seq_len, len(features_cols))
    
    # 4. ПРОГНОЗ И ГЕНЕРАЦИЯ СИГНАЛА
    prediction_scaled = model.predict(input_data)
    predicted_price_normalized = prediction_scaled.flatten()[0]
    
    # Де-нормализация цены
    predicted_price_real = scaler_y.inverse_transform(np.array([[predicted_price_normalized]]))[0][0]
    
    # Текущая цена (последняя в выборке)
    current_price_real = df_processed['close'].iloc[-1]
    
    price_diff = predicted_price_real - current_price_real
    
    if price_diff > 0.00005:
        signal = "BUY"
    elif price_diff < -0.00005:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    # Сохраняем историю (как в predictor_v5.py)
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"{timestamp},{current_price_real},{predicted_price_real},{signal}\n")
        
    return signal, f"✅ Текущая цена: {current_price_real:.5f}"

# --- ИНТЕРФЕЙС STREAMLIT ---

st.title("📊 Forex Analyzer Dashboard v5")
st.write("**Анализатор сигналов для EURUSD M15**")
st.write("---")

# Кнопка обновления данных
if st.button("🔄 Обновить данные и получить сигнал"):
    with st.spinner('⏳ Подключение к терминалу и анализ...'):
        signal, status_msg = get_current_signal()
        
        if signal is None:
            st.error(status_msg) # Выводим ошибку
        else:
            st.success(status_msg)
            
            # Визуализация сигнала
            col1, col2, col3 = st.columns(3)
            colors = {"BUY": "#4CAF50", "SELL": "#F44336", "HOLD": "#FFC107"}
            
            with col1:
                st.write("### Сигнал:")
                st.markdown(f"<h1 style='text-align: center; color: {colors[signal]}'>{signal}</h1>", unsafe_allow_html=True)
            
            with col2:
                st.write("### Статус:")
                st.success("Активно") if signal != "HOLD" else st.warning("Ожидание")
            
            with col3:
                st.write("### Рекомендация:")
                st.info("Покупать") if signal == "BUY" else st.info("Продавать") if signal == "SELL" else st.info("Вне рынка")
                
            st.write("---")
            
            # График истории сигналов
            st.subheader("📈 История прогнозов")
            history_df = load_history_data()
            
            if not history_df.empty:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                color_price = 'tab:blue'
                ax1.set_xlabel('Время')
                ax1.set_ylabel('Цена (EURUSD)', color=color_price)
                ax1.plot(history_df['timestamp'], history_df['price'], color=color_price, label='Цена')
                ax1.tick_params(axis='y', labelcolor=color_price)
                
                ax2 = ax1.twinx()  
                
                color_signal_buy = '#4CAF50'
                color_signal_sell = '#F44336'
                
                ax2.set_ylabel('Сигнал', color='gray')
                ax2.plot(history_df['timestamp'], np.where(history_df['signal']=='BUY', 1.05, np.nan), 
                         'o', color=color_signal_buy, markersize=12, label='BUY')
                ax2.plot(history_df['timestamp'], np.where(history_df['signal']=='SELL', 1.03, np.nan), 
                         'o', color=color_signal_sell, markersize=12, label='SELL')
                
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("История пока пуста. Нажмите кнопку обновления еще раз.")