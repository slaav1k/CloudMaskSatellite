import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

RES = 256
TRAIN = "38-cloud/train/"
IMG = "img/data/"
MASK = "mask/data/"


# Plot the probabilities
def plot(H, S, V):
    X = range(256)
    # Plot smoothened probability graphs for H, S and V
    plt.plot(X, H, color="tab:blue", label="Hue", zorder=3)
    plt.plot(X, S, color="tab:red", label="Saturation", zorder=3, linestyle=(0, (1, 1)))
    plt.plot(X, V, color="tab:green", label="Value", zorder=3, linestyle="dashed")

    plt.ylabel("Probability")
    plt.xlabel("Value of component")
    plt.title("Cloud probability per HSV-component over the train set")

    plt.legend()
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.grid(zorder=0)
    plt.show()


# Process train set and compute probabilities
def process_img():
    print("Computing probabilities over the train set...")

    n_samples = len(os.listdir(TRAIN + IMG))  # Number of samples processed
    H_cloud, S_cloud, V_cloud = np.zeros(256), np.zeros(256), np.zeros(
        256)  # Count cloud-pixels for every possible value
    H_no, S_no, V_no = np.zeros(256), np.zeros(256), np.zeros(256)  # Count nocloud-pixels for every possible value

    with tqdm(total=n_samples) as bar:
        for img_f, mask_f in zip(os.listdir(TRAIN + IMG), os.listdir(TRAIN + MASK)):
            bar.update(1)

            img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, IMG, img_f)), cv2.COLOR_BGR2HSV)
            mask = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, MASK, mask_f)), cv2.COLOR_BGR2GRAY)

            for y in range(RES):
                for x in range(RES):
                    if mask[y][x]:  # Cloud pixel
                        H_cloud[img[y][x][0]] += 1
                        S_cloud[img[y][x][1]] += 1
                        V_cloud[img[y][x][2]] += 1
                    else:  # No cloud pixel
                        H_no[img[y][x][0]] += 1
                        S_no[img[y][x][1]] += 1
                        V_no[img[y][x][2]] += 1

    # Compute probabilities and smoothen them
    H = savgol_filter(H_cloud / np.maximum((H_no + H_cloud), np.ones(256)), 11, 4)
    S = savgol_filter(S_cloud / np.maximum((S_no + S_cloud), np.ones(256)), 11, 4)
    V = savgol_filter(V_cloud / np.maximum((V_no + V_cloud), np.ones(256)), 11, 4)

    np.save("H.npy", H, allow_pickle=True)
    np.save("S.npy", S, allow_pickle=True)
    np.save("V.npy", V, allow_pickle=True)

    return H, S, V  # Probabilities of being a cloud pixel per HSV-component value


DATASET = "RICE1/"
HAZE = "cloud/"
CLEAR = "label/"


def process_img_haze():
    print("Computing probabilities for haze...")

    n_samples = len(os.listdir(DATASET + HAZE))  # Number of samples processed
    H_haze, S_haze, V_haze = np.zeros(256), np.zeros(256), np.zeros(256)  # Count haze-pixels for every possible value
    H_clear, S_clear, V_clear = np.zeros(256), np.zeros(256), np.zeros(
        256)  # Count clear-pixels for every possible value

    with tqdm(total=n_samples) as bar:
        for haze_img, clear_img in zip(os.listdir(DATASET + HAZE), os.listdir(DATASET + CLEAR)):
            bar.update(1)

            # Read and convert images to HSV
            haze = cv2.cvtColor(cv2.imread(os.path.join(DATASET, HAZE, haze_img)), cv2.COLOR_BGR2HSV)
            clear = cv2.cvtColor(cv2.imread(os.path.join(DATASET, CLEAR, clear_img)), cv2.COLOR_BGR2HSV)

            for y in range(haze.shape[0]):
                for x in range(haze.shape[1]):
                    # Assuming we compare haze and clear images directly
                    if np.any(haze[y][x] != clear[y][x]):  # Detect haze pixel
                        H_haze[haze[y][x][0]] += 1
                        S_haze[haze[y][x][1]] += 1
                        V_haze[haze[y][x][2]] += 1
                    else:  # Clear pixel
                        H_clear[clear[y][x][0]] += 1
                        S_clear[clear[y][x][1]] += 1
                        V_clear[clear[y][x][2]] += 1

    # Compute probabilities and smoothen them
    H = savgol_filter(H_haze / np.maximum((H_clear + H_haze), np.ones(256)), 11, 4)
    S = savgol_filter(S_haze / np.maximum((S_clear + S_haze), np.ones(256)), 11, 4)
    V = savgol_filter(V_haze / np.maximum((V_clear + V_haze), np.ones(256)), 11, 4)

    np.save("H_haze.npy", H, allow_pickle=True)
    np.save("S_haze.npy", S, allow_pickle=True)
    np.save("V_haze.npy", V, allow_pickle=True)

    return H, S, V  # Probabilities of being a haze pixel per HSV-component value


# def predict(H, S, V):
#     print("Predicting cloud mask on test set...")
#
#     n_samples = len(os.listdir(TEST + IMG))  # Number of samples processed
#     TP, FP, TN, FN = 0, 0, 0, 0
#
#     with tqdm(total=n_samples) as bar:
#         for img_f, mask_f in zip(os.listdir(TEST + IMG), os.listdir(TEST + MASK)):
#             bar.update(1)
#
#             img = cv2.cvtColor(cv2.imread(os.path.join(TEST, IMG, img_f)), cv2.COLOR_BGR2HSV)
#             mask = cv2.cvtColor(cv2.imread(os.path.join(TEST, MASK, mask_f)), cv2.COLOR_BGR2GRAY) / 255
#
#             pred = np.zeros((RES, RES))
#             for y in range(RES):
#                 for x in range(RES):
#                     h = img[y][x][0]
#                     s = img[y][x][1]
#                     v = img[y][x][2]
#
#                     P = round((H[h] + S[s] + V[v]) / 3)  # Probability to be a cloud, round it to nearest int
#                     pred[y][x] = P
#
#                     if P == 1:  # Prediction -> cloud-pixel
#                         if P == mask[y][x]:
#                             TP += 1
#                         else:
#                             FP += 1
#                     else:  # Prediction -> no-cloud-pixel
#                         if P == mask[y][x]:
#                             TN += 1
#                         else:
#                             FN += 1
#
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     F1 = TP / (TP + 0.5 * (FP + FN))
#     print("Predictions completed! Results:")
#     print(f"Precision: {precision}\nRecall: {recall}\nF1-score: {F1}")

def process_two_images(haze_img_path, clear_img_path):
    """
    Вычисляет вероятности пикселей, принадлежащих к дымке и чистому небу, для двух изображений.

    :param haze_img_path: Путь к изображению с дымкой.
    :param clear_img_path: Путь к изображению без облаков.
    :return: Вероятности H, S, V для пикселей с дымкой.
    """
    print(f"Processing: {haze_img_path} and {clear_img_path}")

    # Гистограммы для накопления значений HSV
    H_haze, S_haze, V_haze = np.zeros(256), np.zeros(256), np.zeros(256)
    H_clear, S_clear, V_clear = np.zeros(256), np.zeros(256), np.zeros(256)

    # Загрузка изображений и преобразование в HSV
    haze_img = cv2.cvtColor(cv2.imread(haze_img_path), cv2.COLOR_BGR2HSV)
    clear_img = cv2.cvtColor(cv2.imread(clear_img_path), cv2.COLOR_BGR2HSV)

    # Проверяем, что размеры совпадают
    if haze_img.shape != clear_img.shape:
        raise ValueError("Размеры изображений не совпадают.")

    # Проходим по всем пикселям
    height, width, _ = haze_img.shape
    for y in range(height):
        for x in range(width):
            haze_pixel = haze_img[y, x]
            clear_pixel = clear_img[y, x]

            # Если пиксели отличаются, считаем это дымкой
            if not np.array_equal(haze_pixel, clear_pixel):
                H_haze[haze_pixel[0]] += 1
                S_haze[haze_pixel[1]] += 1
                V_haze[haze_pixel[2]] += 1
            else:  # Если пиксели совпадают, считаем их чистыми
                H_clear[clear_pixel[0]] += 1
                S_clear[clear_pixel[1]] += 1
                V_clear[clear_pixel[2]] += 1

    # Вычисляем вероятности (гладкие распределения)
    H = savgol_filter(H_haze / np.maximum(H_haze + H_clear, np.ones(256)), 11, 4)
    S = savgol_filter(S_haze / np.maximum(S_haze + S_clear, np.ones(256)), 11, 4)
    V = savgol_filter(V_haze / np.maximum(V_haze + V_clear, np.ones(256)), 11, 4)

    return H, S, V


if __name__ == "__main__":
    # if not os.path.exists("H.npy"):
    # H, S, V = process_img_haze()
    H, S, V = process_two_images("RICE1/cloud/488.png", "RICE1/label/488.png")
    np.save("H_haze_single.npy", H, allow_pickle=True)
    np.save("S_haze_single.npy", S, allow_pickle=True)
    np.save("V_haze_single.npy", V, allow_pickle=True)
    # else:
    #     H = np.load("H.npy", allow_pickle=True)
    #     S = np.load("S.npy", allow_pickle=True)
    #     V = np.load("V.npy", allow_pickle=True)

    plot(H, S, V)
