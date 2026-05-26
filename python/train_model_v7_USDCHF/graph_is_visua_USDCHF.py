import json
import matplotlib.pyplot as plt

# Пробуем загрузить историю из файла
try:
    with open('python/train_model_v7_USDCHF/history_v7_USDCHF.json', 'r') as f:
        history = json.load(f)
    print("✅ История успешно загружена из файла.")

    # --- ГРАФИК ПОТЕРЬ (Loss) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss (MSE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Проверяем, есть ли метрика MAE (если ты компилировал с metrics=['mae'])
    if 'mae' in history and 'val_mae' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Train MAE', color='green')
        plt.plot(history['val_mae'], label='Validation MAE', color='red')
        plt.title('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout() # Чтобы графики не налезали друг на друга
    plt.show()

except FileNotFoundError:
    print("❌ Ошибка: Файл 'history_v7_GBPUSD.json' не найден.")
    print("Убедись, что ячейка с обучением модели была успешно запущена и завершилась без ошибок.")
except Exception as e:
    print(f"⚠️ Произошла ошибка при чтении файла: {e}")