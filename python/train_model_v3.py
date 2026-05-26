# Тут будет регрессия 


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
FUTURE_PERIOD_PREDICT = 3 # Теперь предсказываем цену через 3 свечи (45 минут)
EPOCHS = 20
BATCH_SIZE = 64
# ------------------

def preprocess_regression_data(df):
    """Функция для подготовки данных к задаче регрессии (предсказание цены)"""
    df = df.copy()
    
    # 1. УДАЛЯЕМ ВСЕ НЕНУЖНЫЕ КОЛОНКИ
    df = df.drop(['open', 'datetime', 'pair'], axis=1, errors='ignore')
    
    # 2. СОЗДАЕМ МЕТКУ ДЛЯ РЕГРЕССИИ (Target)
    # Теперь цель - это конкретное значение цены в будущем.
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
    df.dropna(inplace=True)
    
    # Разделяем на признаки (X) и целевую переменную (y)
    X = df.drop('future_close', axis=1).values
    y = df['future_close'].values
    
    # 3. МАСШТАБИРОВАНИЕ (ОЧЕНЬ ВАЖНО ДЛЯ РЕГРЕССИИ)
    # Масштабируем и признаки, и целевую переменную.
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 4. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ДЛЯ LSTM
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in range(len(X_scaled)):
        features = X_scaled[i]
        target = y_scaled[i]
        prev_days.append(features)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), target])
    
    random.shuffle(sequential_data)

    X_final = []
    y_final = []
    
    for seq, target in sequential_data:
        X_final.append(seq)
        y_final.append(target)
        
    return np.array(X_final), np.array(y_final), scaler_y # Возвращаем и масштабер для цены


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
    
    print("Предобработка данных для регрессии...")
    X, y, scaler_y = preprocess_regression_data(main_df)

    print("Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # --- АРХИТЕКТУРА МОДЕЛИ ДЛЯ РЕГРЕССИИ ---
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(64))
    model.add(Dropout(0.3))
    
    model.add(Dense(32))
    
    # ВЫХОДНОЙ СЛОЙ ДЛЯ РЕГРЕССИИ (1 нейрон без активации)
    model.add(Dense(1))

    # КОМПИЛЯЦИЯ МОДЕЛИ (СРЕДНЕКВАДРАТИЧНАЯ ОШИБКА для регрессии)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt) # Метрика accuracy здесь не нужна

    print("Начало обучения модели (Регрессия)...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    val_loss = model.evaluate(X_test, y_test)
    print(f"Среднеквадратичная ошибка на тестовых данных: {val_loss:.4f}")

    # Сохраняем модель и масштабер для цены
    model.save("model_v3.h5")
    
    # Сохраняем масштабер, чтобы потом можно было вернуть цену из нормированного значения
    import joblib
    joblib.dump(scaler_y, 'scaler_price.pkl') 
    
    print("Модель v3 успешно сохранена в файл 'model_v3.h5'")


if __name__ == "__main__":
    import tensorflow as tf
    
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()