import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
H = np.load("H.npy", allow_pickle=True)
H2 = np.load("H_haze.npy", allow_pickle=True)

# Вычисление разницы
difference = H - H2

# Визуализация
plt.figure(figsize=(15, 5))

# График для H
plt.subplot(1, 3, 1)
plt.plot(H, label="H (clouds)")
plt.title("Probabilities - H (clouds)")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.legend()

# График для H2
plt.subplot(1, 3, 2)
plt.plot(H2, label="H2 (haze)", color="orange")
plt.title("Probabilities - H2 (haze)")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.legend()

# График разницы
plt.subplot(1, 3, 3)
plt.plot(difference, label="Difference (H - H2)", color="red")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.title("Difference Between Probabilities")
plt.xlabel("Index")
plt.ylabel("Difference")
plt.legend()

plt.tight_layout()
plt.show()
