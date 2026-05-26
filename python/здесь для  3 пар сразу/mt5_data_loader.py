import os
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5

# --- НАСТРОЙКИ ---
OUTPUT_FOLDER = 'processed_data' # сохранять файлы сюда
PAIRS = ['EURUSD', 'GBPUSD', 'USDCHF']
TIMEFRAME = mt5.TIMEFRAME_M15 
# ------------------

# Инициализируем подключение к терминалу
if not mt5.initialize():
    print("Не удалось подключиться к терминалу MetaTrader 5. Ошибка: ", mt5.last_error())
    quit()

# Получаем текущую дату и отступаем на 3 года назад для истории
today = datetime.today()
free_years_ago = today - timedelta(days=1095) # 3 года

# Создаем папку для выходных данных
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for pair in PAIRS:
    print(f"--- Загрузка данных для {pair} ---")
    
    # Запрашиваем историю (rates)
    rates = mt5.copy_rates_range(pair, TIMEFRAME, free_years_ago, today)
    
    if rates is None or len(rates) == 0:
        print(f"Ошибка при загрузке данных для {pair}. Ошибка: ", mt5.last_error())
        continue

    # Создаем DataFrame из полученных данных
    df = pd.DataFrame(rates)
    
    # Преобразуем время из Unix-формата в понятный вид
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Переименуем колонки для красоты и удобства
    df.rename(columns={
        'time': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'spread': 'spread'
    }, inplace=True)
    
    # Оставим только нужные колонки
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Получено строк: {len(df)}")
    
    # Сохраняем в CSV
    output_path = os.path.join(OUTPUT_FOLDER, f'{pair}_M15.csv')
    df.to_csv(output_path, index=False)
    print(f"Данные сохранены в {output_path}\n")

# Завершаем соединение с терминалом
mt5.shutdown()