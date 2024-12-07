import cv2
import numpy as np
import matplotlib.pyplot as plt


# def get_mask(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     height, width = image.shape
#     num_regions = 10
#     region_height = height // num_regions
#
#     # Применить CLAHE (Contrast Limited Adaptive Histogram Equalization) для каждой области
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     regions_equalized = image.copy()
#
#     for i in range(num_regions):
#         region = regions_equalized[i * region_height:(i + 1) * region_height, :]
#         regions_equalized[i * region_height:(i + 1) * region_height, :] = clahe.apply(region)
#
#     # глобальное выравнивание на все изображение
#     equalized_image = clahe.apply(regions_equalized)
#     equalized_image = clahe.apply(equalized_image)
#
#     # гистограммное выравнивание
#     hist_equalized_image = cv2.equalizeHist(image)
#     hist_equalized_image = cv2.equalizeHist(hist_equalized_image)
#
#     # пороговое значение для выделения облаков
#     threshold_value = 130
#     _, cloud_mask = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)
#
#     # Показать результаты
#     plt.figure(figsize=(50, 150))
#
#     # Оригинальное изображение
#     plt.subplot(1, 3, 1)
#     plt.title('Оригинальное изображение')
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.imsave('orig_img.bmp', image, cmap='gray')
#
#     # Уравненное изображение по регионам
#     plt.subplot(1, 3, 2)
#     plt.title('Уравненное изображение по регионам')
#     plt.imshow(regions_equalized, cmap='gray')
#     plt.axis('off')
#     plt.imsave('regions_equalized.bmp', regions_equalized, cmap='gray')
#
#     # Маска облаков
#     plt.subplot(1, 3, 3)
#     plt.title('Маска облаков')
#     plt.imshow(cloud_mask, cmap='gray')
#     plt.axis('off')
#     plt.imsave('cloud_mask.bmp', cloud_mask, cmap='gray')
#     plt.show()


def get_mask2(image_path):
    # Загружаем изображение без изменения битности и формата
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Ошибка: файл '{image_path}' не найден или не поддерживается.")
        return

    height, width = image.shape
    num_regions = 10
    region_height = height // num_regions

    # Применить CLAHE (Contrast Limited Adaptive Histogram Equalization) для каждой области
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    regions_equalized = image.copy()

    for i in range(num_regions):
        region = regions_equalized[i * region_height:(i + 1) * region_height, :]
        regions_equalized[i * region_height:(i + 1) * region_height, :] = clahe.apply(region)

    # глобальное выравнивание на все изображение
    equalized_image = clahe.apply(regions_equalized)
    equalized_image = clahe.apply(equalized_image)

    # гистограммное выравнивание для 16 бит не работает
    #     hist_equalized_image = cv2.equalizeHist(image)
    #     hist_equalized_image = cv2.equalizeHist(hist_equalized_image)

    # # пороговое значение для выделения облаков
    # threshold_value = 27000
    # _, cloud_mask = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Разделим изображение на верхнюю и нижнюю половину
    upper_half = equalized_image[:height // 2, :]
    lower_half = equalized_image[height // 2:, :]

    # Для верхней половины используем один порог
    threshold_value_upper = 30000
    _, cloud_mask_upper = cv2.threshold(upper_half, threshold_value_upper, 255, cv2.THRESH_BINARY)

    # Для нижней половины используем другой порог
    threshold_value_lower = 25000
    _, cloud_mask_lower = cv2.threshold(lower_half, threshold_value_lower, 255, cv2.THRESH_BINARY)

    # Собираем маски обратно
    cloud_mask = np.vstack((cloud_mask_upper, cloud_mask_lower))


    # Показать результаты
    plt.figure(figsize=(50, 150))

    # Оригинальное изображение
    plt.subplot(1, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.imsave('orig_img.bmp', image, cmap='gray')

    # Уравненное изображение по регионам
    plt.subplot(1, 3, 2)
    plt.title('Уравненное изображение по регионам')
    plt.imshow(regions_equalized, cmap='gray')
    plt.axis('off')
    plt.imsave('regions_equalized.bmp', regions_equalized, cmap='gray')

    # Маска облаков
    plt.subplot(1, 3, 3)
    plt.title('Маска облаков')
    plt.imshow(cloud_mask, cmap='gray')
    plt.axis('off')
    plt.imsave('cloud_mask.bmp', cloud_mask, cmap='gray')
    plt.show()


get_mask2("../201707011400_A1_04.tif")
# get_mask("201707012000_A1_04.bmp")
