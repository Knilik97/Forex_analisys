import os
import pandas as pd

# --- НАСТРОЙКИ ---
INPUT_FOLDER = 'final_data'  # Папка с данными, где уже есть индикаторы
OUTPUT_FOLDER = 'ml_ready_data' # Новая папка для финальных данных
LOOKAHEAD_STEPS = 3 # Смотрим вперед на 3 свечи (45 минут), чтобы определить цель
# ------------------

def add_target_column(df):
    """Функция добавляет колонку 'target' в DataFrame"""
    # Создаем серию, сдвинутую назад на LOOKAHEAD_STEPS шагов
    future_close = df['close'].shift(-LOOKAHEAD_STEPS)
    
    # Сравниваем текущую цену закрытия с будущей
    # Если будущая цена > текущей, то target = 1, иначе 0
    df['target'] = (future_close > df['close']).astype(int)
    
    # Удаляем последние LOOKAHEAD_STEPS строк, так как для них нет будущего значения
    df = df.iloc[:-LOOKAHEAD_STEPS]
    
    return df

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('_features.csv'):
            pair_name = filename.split('_')[0]
            print(f"--- Обработка файла: {filename} ---")
            
            input_path = os.path.join(INPUT_FOLDER, filename)
            df = pd.read_csv(input_path)
            
            if not df.empty:
                df = add_target_column(df)
                
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                df.to_csv(output_path, index=False)
                
                print(f"Добавлены метки. Файл сохранен в {output_path}")
                print(f"Строк в файле: {len(df)}")
            else:
                print(f"Файл {filename} пуст.")

if __name__ == "__main__":
    main()