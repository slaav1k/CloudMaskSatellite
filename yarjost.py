import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузить изображение
image_path = 'C:/Users/arkhi/Desktop/screens/3.jpg'  # укажите путь к изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Разделить изображение на области (например, по горизонтали)
height, width = image.shape
num_regions = 10
region_height = height // num_regions

# Применить CLAHE (Contrast Limited Adaptive Histogram Equalization) для каждой области
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
regions_equalized = image.copy()

image2 = clahe.apply(image)

for i in range(num_regions):
    region = regions_equalized[i * region_height:(i + 1) * region_height, :]
    regions_equalized[i * region_height:(i + 1) * region_height, :] = clahe.apply(region)

# После этого можно применить глобальное выравнивание на все изображение
equalized_image = clahe.apply(regions_equalized)

# Применить стандартное гистограммное выравнивание
hist_equalized_image = cv2.equalizeHist(image)

# Применить пороговое значение для выделения облаков
threshold_value = 130
_, cloud_mask = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)

# Морфологическая обработка
kernel = np.ones((5, 5), np.uint8)
cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)


# Применить дилатацию для расширения маски (умеренно)
mask_dilated = cv2.dilate(cloud_mask, None, iterations=1)

# Используем inpainting для удаления облаков
inpainted_image = cv2.inpaint(regions_equalized, mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Показать результаты
plt.figure(figsize=(50, 10))

# Оригинальное изображение
plt.subplot(1, 5, 1)
plt.title('Оригинальное изображение')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Простое выравнивание
plt.subplot(1, 5, 2)
plt.title('Простое выравнивание')
plt.imshow(image2, cmap='gray')
plt.axis('off')

# Простое выравнивание по регионам
plt.subplot(1, 5, 3)
plt.title('Выравнивание по регионам')
plt.imshow(hist_equalized_image, cmap='gray')
plt.axis('off')

# Уравненное изображение по регионам
plt.subplot(1, 5, 4)
plt.title('Уравненное изображение по регионам')
plt.imshow(regions_equalized, cmap='gray')
plt.axis('off')

# Маска облаков
plt.subplot(1, 5, 5)
plt.title('Маска облаков')
plt.imshow(cloud_mask, cmap='gray')
plt.axis('off')

plt.show()

# Показать изображение после удаления облаков
plt.figure(figsize=(10, 10))
plt.imshow(inpainted_image, cmap='gray')
plt.title('Изображение после удаления облаков')
plt.axis('off')
plt.show()
