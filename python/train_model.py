import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
import random
import time

# --- НАСТРОЙКИ ---
SEQ_LEN = 60            # Сколько прошлых свечей модель будет "помнить" для прогноза
FUTURE_PERIOD_PREDICT = 3 # На сколько свечей вперед мы предсказываем (должно совпадать с create_labels.py)
EPOCHS = 10              # Сколько раз модель "пройдет" по всем данным
BATCH_SIZE = 64          # Размер порции данных для обучения за один шаг
# ------------------

def classify(current, future):
    """Функция для создания метки (столбец 'target')"""
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    """Функция для предобработки данных перед подачей в нейросеть"""
    print(f"--- Начало предобработки. Размер данных: {len(df)} ---")
    df = df.copy()
    
    # --- НОВОЕ: Убираем колонку 'pair' в самом начале ---
    # Она содержит текст ("EURUSD"), который ломает математику модели.
    df = df.drop(['pair'], axis=1, errors='ignore') # errors='ignore' на случай, если колонки нет
    # --- КОНЕЦ НОВОГО КОДА ---
    
    df = df.drop(['open', 'datetime'], axis=1)
    
    for col in df.columns:
        if col not in ["target"]: # Теперь нам не нужно проверять 'pair', мы его уже удалили
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    print(f"Итоговый размер данных после всех преобразований: {len(df)} строк")

    if len(df) == 0:
        print("ОШИБКА: Данные закончились!")
        return np.array([]), np.array([])

    if len(df) < SEQ_LEN:
        print(f"ОШИБКА: Данных ({len(df)}) меньше, чем длина последовательности ({SEQ_LEN}).")
        return np.array([]), np.array([])

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        features = i[:-1] 
        target = i[-1]
        prev_days.append(features)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), target])
    
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

    # Собираем данные по всем валютным парам в один большой DataFrame
    for filename in os.listdir(datasets_path):
        if filename.endswith('_features.csv'):
            pair = filename.split('_')[0]
            print(f"Обрабатываю пару: {pair}")
            df = pd.read_csv(os.path.join(datasets_path, filename))
            
            # Добавляем название пары как колонку (это поможет модели понять, что пары разные)
            df['pair'] = pair
            
            if len(main_df) == 0:
                main_df = df
            else:
                main_df = pd.concat([main_df, df])

    main_df.dropna(inplace=True)
    
    print("Предобработка данных...")
    X, y = preprocess_df(main_df)

    print("Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Не перемешиваем время!

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # --- СОЗДАНИЕ МОДЕЛИ LSTM ---
    model = Sequential()

    model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(32))
    model.add(Dropout(0.2))
    
    model.add(Dense(2, activation="softmax")) # 2 выхода: Покупка или Продажа

    # Компиляция модели
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))

    print("Начало обучения модели...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        # callbacks=[tensorboard] # Уберу данный параметр из-за отсутсвия 
    )

    # Оценка модели
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Точность на тестовых данных: {val_acc:.4f}")

    # Сохраняем обученную модель
    model.save("model.h5")
    print("Модель успешно сохранена в файл 'model.h5'")


if __name__ == "__main__":
    import tensorflow as tf
    from sklearn import preprocessing
    
    # Фиксируем сид для воспроизводимости результатов
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    main()