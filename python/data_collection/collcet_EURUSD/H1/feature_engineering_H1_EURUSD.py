import pandas as pd
import numpy as np

# --- НАСТРОЙКИ ---
INPUT_FILE = "data/EUR_USD/H1/EURUSD_H1.csv"
OUTPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_features.csv"
# ------------------

print("🔧 Запуск расчета технических индикаторов...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Данные загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить data_collector.py")
    quit()

# --- РАСЧЕТ ИНДИКАТОРОВ ---
# Используем колонку 'close' как базу для большинства индикаторов
close_prices = df['close']

# 1. Скользящие средние (MA)
df['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()

# 2. Индекс относительной силы (RSI)
delta = close_prices.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan) # Защита от деления на ноль в начале
df['RSI'] = 100 - (100 / (1 + rs))

# 3. Схождение/расхождение скользящих средних (MACD)
EMA_12 = close_prices.ewm(span=12, adjust=False).mean()
EMA_26 = close_prices.ewm(span=26, adjust=False).mean()
df['MACD'] = EMA_12 - EMA_26 
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Histogram'] = df['MACD'] - df['Signal']

# --- ОЧИСТКА ДАННЫХ ---
# Индикаторы в начале DataFrame будут содержать NaN, так как им не хватает данных для расчета.
# Удаляем эти строки.
df = df.dropna()

print(f"✅ Индикаторы рассчитаны. Удалено строк с NaN. Осталось: {len(df)}")

# --- СОХРАНЕНИЕ ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ УСПЕХ! Данные с индикаторами сохранены в {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")