import os
import pandas as pd
import numpy as np

# --- НАСТРОЙКИ ---
INPUT_FOLDER = 'processed_data'  # Папка с исходными данными (M15)
OUTPUT_FOLDER = 'final_data'     # Папка для готовых данных с индикаторами
PAIRS = ['EURUSD', 'GBPUSD', 'USDCHF']
# ------------------

# Функция для расчета RSI (Индекс относительной силы)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    delta = delta[1:] # Убираем первое NaN значение

    # Делим прибыль и убытки
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Рассчитываем среднее (EWMA - экспоненциальное скользящее)
    roll_up = up.ewm(com=(window - 1), min_periods=window).mean()
    roll_down = down.abs().ewm(com=(window - 1), min_periods=window).mean()

    # Рассчитываем RS и RSI
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Добавляем NaN в начало, чтобы длина серии совпала с исходной
    rsi = pd.concat([pd.Series([np.nan]), rsi])
    return rsi

# Функция для расчета MACD
def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def main():
    # Создаем папку для выходных данных, если ее нет
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for pair in PAIRS:
        input_path = os.path.join(INPUT_FOLDER, f'{pair}_M15.csv')
        
        if not os.path.exists(input_path):
            print(f"Файл не найден: {input_path}. Пропускаем.")
            continue

        print(f"--- Обработка пары: {pair} ---")
        
        # Читаем CSV-файл
        df = pd.read_csv(input_path)
        
        if df.empty:
            print(f"Файл {pair}_M15.csv пуст.")
            continue

        # --- ГЕНЕРАЦИЯ ПРИЗНАКОВ ---
        
        # 1. Скользящие средние (MA)
        df['MA_16'] = df['close'].rolling(window=16).mean()   # Короткая MA
        df['MA_30'] = df['close'].rolling(window=30).mean()   # Длинная MA
        
        # 2. RSI (Индекс относительной силы)
        df['RSI_14'] = calculate_rsi(df['close'])
        
        # 3. MACD (Схождение/расхождение скользящих средних)
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        df['MACD_Line'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Hist'] = histogram

        # Сохраняем результат в новую папку
        output_path = os.path.join(OUTPUT_FOLDER, f'{pair}_M15_features.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Добавлены индикаторы. Файл сохранен в {output_path}")
        print(f"Строк в файле: {len(df)}")
        print("-" * 30)

if __name__ == "__main__":
    main()