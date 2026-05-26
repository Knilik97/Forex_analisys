import os
import pandas as pd
import numpy as np
import random
import time
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- НАСТРОЙКИ ---
SEQ_LEN = 60            # Сколько прошлых свечей модель будет "помнить"
EPOCHS = 20             # Увеличим количество эпох для лучшего обучения
BATCH_SIZE = 64
# ------------------

def preprocess_df(df):
    """Улучшенная функция предобработки данных"""
    df = df.copy()
    
    # 1. УДАЛЯЕМ ВСЕ НЕНУЖНЫЕ КОЛОНКИ (включая 'pair')
    df = df.drop(['open', 'datetime', 'pair'], axis=1, errors='ignore')
    
    # 2. СОЗДАЕМ МЕТКУ КЛАССИФИКАЦИИ (Target)
    # Мы будем предсказывать движение цены на следующую свечу (FUTURE_PERIOD_PREDICT = 1)
    df['target'] = df['close'].shift(-1)
    df['target'] = (df['target'] > df['close']).astype(int) # 1 если цена выросла, 0 если упала
    df.dropna(inplace=True)
    
    # 3. УЛУЧШЕННОЕ МАСШТАБИРОВАНИЕ (Scaling)
    # Масштабируем все данные ОДНОВРЕМЕННО, а не по колонкам.
    # Это сохраняет взаимосвязи между параметрами.
    scaler = StandardScaler()
    
    # Применяем масштабирование ко всем колонкам, кроме 'target'
    columns_to_scale = df.columns.drop('target')
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # 4. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ДЛЯ LSTM
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        features = i[:-1] 
        target = i[-1]
        prev_days.append(features)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), target])
    
    # --- УЛУЧШЕННАЯ БАЛАНСИРОВКА КЛАССОВ ---
    # Вместо обрезки до минимума, мы просто перемешаем и разделим.
    # Это сохранит больше данных для обучения.
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), np.array(y)


def main():
    print("Загрузка данных...")
    main_df = pd.DataFrame()
    datasets_path = 'ml_ready_data'

    for filename in os.listdir(datasets_path):
        if filename.endswith('_features.csv'):
            pair = filename.split('_')[0]
            print(f"Обрабатываю пару: {pair}")
            df = pd.read_csv(os.path.join(datasets_path, filename))
            if len(main_df) == 0:
                main_df = df
            else:
                main_df = pd.concat([main_df, df])

    main_df.dropna(inplace=True)
    
    print("Предобработка данных...")
    X, y = preprocess_df(main_df)

    print("Разделение на обучающую и тестовую выборки...")
    # Важно: НЕ перемешиваем время! Временные ряды нельзя мешать.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # --- УЛУЧШЕННАЯ АРХИТЕКТУРА МОДЕЛИ ---
    model = Sequential()
    
    # Входной слой LSTM (помнит SEQ_LEN свечей)
    model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2)) # Dropout для борьбы с переобучением
    
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(64)) # Добавили еще один плотный слой для сложности
    model.add(Dropout(0.2))
    
    model.add(Dense(32))
    model.add(Dropout(0.2))
    
    # ВЫХОДНОЙ СЛОЙ ДЛЯ БИНАРНОЙ КЛАССИФИКАЦИИ (1 нейрон с сигмоидой)
    model.add(Dense(1, activation='sigmoid'))

    # КОМПИЛЯЦИЯ МОДЕЛИ (БИНАРНАЯ КРОССЭНТРОПИЯ для да/нет)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("Начало обучения модели...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1 # Показывать прогресс в терминале красиво
    )

    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Точность на тестовых данных: {val_acc:.4f}")

    # Сохраняем улучшенную модель под новым именем
    model.save("model_v2.h5")
    print("Улучшенная модель успешно сохранена в файл 'model_v2.h5'")


if __name__ == "__main__":
    import tensorflow as tf
    
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()