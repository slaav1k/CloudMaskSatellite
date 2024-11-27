import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузим вероятности для каждого компонента
H = np.load("H.npy", allow_pickle=True)
S = np.load("S.npy", allow_pickle=True)
V = np.load("V.npy", allow_pickle=True)

# Параметры
RES = 256  # Разрешение изображения


def predict_cloud_mask(image_path, H, S, V):
    # Чтение изображения, в формате HSV
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)

    mask_pred = np.zeros(img.shape[:2], dtype=np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            h = img[y][x][0]
            s = img[y][x][1]
            v = img[y][x][2]

            # Вычисление вероятности облака на основе компонент HSV
            P = round((H[h] + S[s] + V[v]) / 3)

            mask_pred[y][x] = 255 if P == 1 else 0

    return mask_pred


def remove_clouds(image, mask):
    mask_dilated = cv2.dilate(mask, None, iterations=5)
    # Заполнение пустоты интерполяцией
    result = cv2.inpaint(image, mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result


def remove_clouds_without_inpainting(image, mask):
    # Убираем облака, но без интерполяции (просто заменяем на чёрный цвет)
    image_without_inpainting = image.copy()
    image_without_inpainting[mask == 255] = [0, 0, 0]  # Заменяем облачные пиксели на чёрные
    return image_without_inpainting


# Путь к изображению, для которого нужно выделить маску облаков
image_path = '07WER_10000.jpg'

# Предсказание маски облаков
cloud_mask = predict_cloud_mask(image_path, H, S, V)

# Чтение исходного изображения
image = cv2.imread(image_path)

# Убираем облака с помощью интерполяции (inpainting)
image_without_clouds = remove_clouds(image, cloud_mask)

# Убираем облака без интерполяции, просто заменяя на чёрный цвет
image_without_clouds_no_inpainting = remove_clouds_without_inpainting(image, cloud_mask)

# Отображаем исходное изображение, маску и результат
plt.figure(figsize=(20, 5))

# Отображаем исходное изображение
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Отображаем маску облаков
plt.subplot(1, 4, 2)
plt.imshow(cloud_mask, cmap='gray')
plt.title("Cloud Mask")
plt.axis('off')

# Отображаем изображение без облаков без интерполяции
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(image_without_clouds_no_inpainting, cv2.COLOR_BGR2RGB))
plt.title("Image Without Clouds (No Inpainting)")
plt.axis('off')

# Отображаем изображение без облаков с интерполяцией
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(image_without_clouds, cv2.COLOR_BGR2RGB))
plt.title("Image Without Clouds (Inpainting)")
plt.axis('off')

plt.show()
