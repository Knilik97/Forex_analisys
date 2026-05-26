import pandas as pd

# --- НАСТРОЙКИ ---
input_file = r'data\EUR_USD\M30\EURUSD_M30_full.csv'
output_file = r'data\EUR_USD\M30\EURUSD_M30_markup.csv'

# --- КОД ---
# 1. Читаем файл с явным указанием имен колонок
try:
    df = pd.read_csv(input_file)
    if not all(col in df.columns for col in ['date', 'time', 'close']):
        df = pd.read_csv(input_file, 
                        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
                        header=None)
except Exception as e:
    df = pd.read_csv(input_file, 
                    names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
                    header=None)

# 2. СОЗДАЕМ НОВУЮ КОЛОНКУ 'datetime'
# Меняем формат на тот, который в твоем файле: ГГГГ.ММ.ДД ЧЧ:ММ
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')

# 3. Сортируем данные по времени (критически важно для LSTM!)
df = df.sort_values('datetime').reset_index(drop=True)

# 4. Удаляем старые колонки
df = df.drop(['date', 'time'], axis=1)

# 5. Переносим колонку 'datetime' в начало
cols = list(df.columns)
cols.insert(0, cols.pop(cols.index('datetime')))
df = df[cols]

# 6. Сохраняем чистый файл
df.to_csv(output_file, index=False)

print(f"✅ Данные успешно обработаны и сохранены в: {output_file}")
print(f"✅ Колонки 'date' и 'time' удалены. Используем только 'datetime'.")