import pandas as pd
import numpy as np

# --- НАСТРОЙКИ ---
# ВХОД: Файл, обработанный power_feature_engineering.py
INPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_power_features.csv"
# ВЫХОД: Готовый файл для обучения модели классификации
OUTPUT_FILE = "data/EUR_USD/H1/ready_for_training_EURUSD_H1_Class.csv"

# На сколько свечей вперед смотрим?
FUTURE_PERIODS = 3 
# Порог в пунктах (pips), чтобы считать движение значимым
THRESHOLD_PIPS = 20.0 # Если цена изменилась меньше чем на 20 пунктов - это "Боковой тренд"
# ------------------

print(f"🏷️ Создание целевой переменной (КЛАССИФИКАЦИЯ) на {FUTURE_PERIODS} свечей вперед...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Данные с индикаторами и признаками силы загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить power_feature_engineering.py")
    quit()

# --- СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (КЛАССА) ---
current_close = df['close']
future_close = df['close'].shift(-FUTURE_PERIODS)

# Считаем разницу в пунктах (pips)
df['price_diff_pips'] = (future_close - current_close) * 100000

# Создаем метки классов
labels = []
for diff in df['price_diff_pips']:
    if pd.isna(diff):
        labels.append(np.nan)
    elif diff > THRESHOLD_PIPS:
        labels.append(2) # Класс 2: Сильный рост (BUY)
    elif diff < -THRESHOLD_PIPS:
        labels.append(0) # Класс 0: Сильное падение (SELL)
    else:
        labels.append(1) # Класс 1: Боковой тренд/незначительное изменение (HOLD)

df['target_class'] = labels

# --- ОЧИСТКА ДАННЫХ ---
df.dropna(inplace=True) # Удаляем строки, где не хватило данных для расчета

print(f"✅ Целевая переменная (классы) создана. Удалено строк с NaN.")
print(f"🚀 УСПЕХ! Файл для обучения готов: {OUTPUT_FILE}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Итоговая форма данных: {df.shape}")