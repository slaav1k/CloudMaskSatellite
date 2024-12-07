import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_mask2(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Ошибка: файл '{image_path}' не найден или не поддерживается.")
        return

    height, width = image.shape
    center_y, center_x = height // 2, width // 2  # Центр изображения

    # Фильтрация черного цвета (пиксели, где яркость == 0, считаются черным)
    mask_non_black = image > 0  # Маска для всех пикселей, которые не черные

    # Вычисление средней яркости изображения без черных областей
    total_mean_brightness = np.mean(image[mask_non_black])

    # Количество колец (можно увеличить, чтобы более равномерно покрыть изображение)
    num_rings = 10
    # Увеличиваем радиус колец, чтобы они покрыли все изображение
    ring_height = min(height, width) // (2 * num_rings)

    # Копируем изображение для работы
    image_equalized = image.copy()

    max_intensity = np.iinfo(image.dtype).max

    # Применяем выравнивание по кольцам с прогресс-баром
    for i in tqdm(range(num_rings), desc="Обработка колец", ncols=100):  # Добавляем прогресс-бар
        # Индексы пикселей, принадлежащих кольцу
        inner_radius = i * ring_height
        outer_radius = (i + 1) * ring_height

        for y in range(height):
            for x in range(width):
                # Проверка, что пиксель находится внутри круга
                distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                if inner_radius <= distance_from_center < outer_radius and distance_from_center < min(center_x,
                                                                                                      center_y):
                    # Средняя яркость кольца (по всем пикселям кольца, которые не черные)
                    ring_mean_brightness = np.mean(image[max(0, y - ring_height):y + ring_height,
                                                   max(0, x - ring_height):x + ring_height][
                                                       mask_non_black[max(0, y - ring_height):y + ring_height,
                                                       max(0, x - ring_height):x + ring_height]])

                    # Коэффициент для выравнивания яркости кольца
                    coefficient = total_mean_brightness / ring_mean_brightness if ring_mean_brightness != 0 else 1
                    image_equalized[y, x] = np.clip(image[y, x] * coefficient, 0, max_intensity)

    # Пороговое значение для выделения облаков
    threshold_value = 240
    _, cloud_mask = cv2.threshold(image_equalized, threshold_value, 255, cv2.THRESH_BINARY)

    _, cloud_mask2 = cv2.threshold(image_equalized, 500, 255, cv2.THRESH_BINARY)
    cv2.imwrite('cloud_mask500.tif', cloud_mask2)
    _, cloud_mask2 = cv2.threshold(image_equalized, 700, 255, cv2.THRESH_BINARY)
    cv2.imwrite('cloud_mask700.tif', cloud_mask2)
    _, cloud_mask2 = cv2.threshold(image_equalized, 1000, 255, cv2.THRESH_BINARY)
    cv2.imwrite('cloud_mask1000.tif', cloud_mask2)

    # Показать результаты
    plt.figure(figsize=(50, 150))

    # Оригинальное изображение
    plt.subplot(1, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.imsave('orig_img.bmp', image, cmap='gray')

    # Выравненное изображение по кольцам
    plt.subplot(1, 3, 2)
    plt.title('Выравненное изображение по кольцам')
    plt.imshow(image_equalized, cmap='gray')
    plt.axis('off')
    plt.imsave('image_equalized.bmp', image_equalized, cmap='gray')
    cv2.imwrite('image_equalized.tif', image_equalized)

    # Маска облаков
    plt.subplot(1, 3, 3)
    plt.title('Маска облаков')
    plt.imshow(cloud_mask, cmap='gray')
    plt.axis('off')
    plt.imsave('cloud_mask.bmp', cloud_mask, cmap='gray')
    cv2.imwrite('cloud_mask.tif', cloud_mask)

    plt.show()


def get_only_mask(image_path):
    # Пороговое значение для выделения облаков
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Ошибка: файл '{image_path}' не найден или не поддерживается.")
        return

    threshold_value = 300
    _, cloud_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Показать результаты
    plt.figure(figsize=(50, 150))

    # Выравненное изображение по кольцам
    plt.subplot(1, 2, 1)
    plt.title('Выравненное изображение по кольцам')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.imsave('image_equalized.bmp', image, cmap='gray')
    cv2.imwrite('image_equalized.tif', image)

    # Маска облаков
    plt.subplot(1, 2, 2)
    plt.title('Маска облаков')
    plt.imshow(cloud_mask, cmap='gray')
    plt.axis('off')
    plt.imsave('cloud_mask.bmp', cloud_mask, cmap='gray')
    cv2.imwrite('cloud_mask.tif', cloud_mask)

    plt.show()


# get_mask2("201707011900_A1_04.tif")
get_only_mask("image_equalized.tif")
# get_mask2("cropped_image.tif")
