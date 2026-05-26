import os
import pandas as pd
import MetaTrader5 as mt5

# --- НАСТРОЙКИ ---
SYMBOL = "EURUSD"      # Только одна пара, как ты и хотел
TIMEFRAME = mt5.TIMEFRAME_M30 # _М... данный таймфрейм 
OUTPUT_FOLDER = 'data/EUR_USD/M30' # Папка для сохранения
NUM_CANDLES = 99999   # Количество свечей 
# ------------------

print("📥 Запуск загрузки данных из MT5...")

# Инициализируем подключение к терминалу
if not mt5.initialize():
    print("❌ Не удалось подключиться к терминалу MetaTrader 5.")
    quit()

# Проверяем, есть ли вообще история по этой паре и таймфрейму
if not mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1):
    print(f"❌ Ошибка: Нет данных для {SYMBOL} {TIMEFRAME}. Проверь в терминале: Сервис -> Настройки -> Графики -> Макс. баров в окне.")
    mt5.shutdown()
    quit()

# Создаем папку для выходных данных
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, f'{SYMBOL}_M30.csv') # Здесь меняем название в окочании H1, M15, M30 и и т.п

print(f"🟢 Подключение установлено. Собираем {NUM_CANDLES} свечей для {SYMBOL}...")

# --- СКАЧИВАНИЕ ДАННЫХ ---
# Используем самый надежный метод: от текущей позиции (0) назад на NUM_CANDLES свечей
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

mt5.shutdown()

if rates is None or len(rates) == 0:
    print("❌ Ошибка: Не удалось получить котировки. Проверь историю в терминале.")
    quit()
elif len(rates) < 1000:
    print(f"⚠️  Внимание: Получено всего {len(rates)} свечей. Обучение будет некачественным.")
    input("Нажми Enter, чтобы продолжить, или Ctrl+C для отмены...")

# --- СОХРАНЕНИЕ В CSV ---
df = pd.DataFrame(rates)

# Преобразуем время и переименуем колонки
df['time'] = pd.to_datetime(df['time'], unit='s')
df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)
df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

df.to_csv(output_path, index=False)

print(f"✅ УСПЕХ! Данные сохранены в {output_path}")
print(f"Форма данных: {df.shape}")