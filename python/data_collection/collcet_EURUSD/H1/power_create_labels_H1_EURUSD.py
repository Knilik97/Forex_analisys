import pandas as pd

# --- НАСТРОЙКИ ---
# ВХОД: Файл, обработанный скриптом power_feature_engineering.py
INPUT_FILE = "data/EUR_USD/H1/EURUSD_H1_with_power_features.csv"
# ВЫХОД: Готовый файл для обучения модели
OUTPUT_FILE = "data/EUR_USD/H1/ready_for_training_EURUSD_H1_Power.csv"

# На сколько свечей вперед мы хотим предсказать цену?
# 1 - следующая свеча, 24 - следующий день (на H1), 120 - неделя.
FUTURE_PERIODS = 3 
# ------------------

print(f"🏷️ Создание целевой переменной (Label) на {FUTURE_PERIODS} свечей вперед...")
print(f"📥 Входной файл: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Данные с индикаторами и признаками силы загружены. Количество строк: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
    print("Сначала нужно запустить power_feature_engineering.py")
    quit()

# --- СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (TARGET) ---
# Модель должна предсказать цену 'close' через N свечей вперед.
# Мы сдвигаем колонку 'close' на N шагов вниз.
df['target'] = df['close'].shift(-FUTURE_PERIODS)

# --- ОЧИСТКА ДАННЫХ ---
# После сдвига последние N строк будут содержать NaN в колонке 'target'.
# Удаляем их, так как для них нет "будущей" цены.
initial_rows = len(df)
df = df.dropna()
dropped_rows = initial_rows - len(df)

print(f"✅ Целевая переменная создана. Удалено последних {dropped_rows} строк с NaN.")
print(f"➖ Это нормально, так как для последних {FUTURE_PERIODS} свечей нет данных о будущем.")
print(f"➕ Осталось строк для обучения: {len(df)}")

# --- СОХРАНЕНИЕ ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"🚀 УСПЕХ! Файл для обучения готов: {OUTPUT_FILE}")
print(f"Итоговая форма данных: {df.shape}")