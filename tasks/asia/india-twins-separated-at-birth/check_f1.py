"""
check_f1.py — оценить main.py на обучающей выборке так же, как скрытый проверяющий оценивает тестовую.
Только для разработки (не отправляется на платформу).

ЗАПУСК (из этой папки)
    python check_f1.py            # F1 с настройками из main.py, с достройкой групп и без
    python check_f1.py --sweep    # дополнительно перебрать другие комбинации порогов (замеры считаются только один раз)

КАК СЧИТАЕТСЯ ОЦЕНКА (пример)
    истинных пар (train-output.csv): 1399      наших пар: 1460      верных (есть в обоих): 1345
    precision (точность) = 1345 / 1460 = 0.921   — какая доля НАШИХ пар правильная
    recall (полнота)     = 1345 / 1399 = 0.961   — какую долю ИСТИННЫХ пар мы нашли
    F1 = 2PR / (P+R) = 0.941                     — среднее гармоническое: высокое, только если высоки обе
    Пары неупорядоченные: (a, b) == (b, a), поэтому обе стороны хранятся отсортированными.
"""

import csv
import itertools
import sys
import time
from pathlib import Path

TASK_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_FOLDER))  # чтобы `import main` работал даже при запуске python -I
import main  # noqa: E402

TRAIN_IMAGES_FOLDER = TASK_FOLDER / "data" / "task_india" / "train"
TRAIN_TRUE_PAIRS_FILE = TASK_FOLDER / "data" / "task_india" / "train-output.csv"
PASS_THRESHOLD_F1 = 0.88  # порог зачёта из условия задачи


def read_pairs(csv_path):
    """Прочитать CSV с парами в множество отсортированных кортежей (имя_a, имя_b)."""
    with open(csv_path) as csv_file:
        return {tuple(sorted((row["image_a"], row["image_b"]))) for row in csv.DictReader(csv_file)}


def score(predicted_pairs, true_pairs):
    """Вернуть (precision, recall, F1). Пересечение множеств (&) = верно найденные пары."""
    predicted_pairs = set(predicted_pairs)
    correct_count = len(predicted_pairs & true_pairs)
    precision = correct_count / len(predicted_pairs) if predicted_pairs else 0.0
    recall = correct_count / len(true_pairs) if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def print_score(label, predicted_pairs, true_pairs):
    precision, recall, f1 = score(predicted_pairs, true_pairs)
    verdict = "ПРОЙДЕНО" if f1 >= PASS_THRESHOLD_F1 else "НЕ ПРОЙДЕНО"
    print(f"{label:38} пар={len(set(predicted_pairs)):5}  точность={precision:.4f}  полнота={recall:.4f}  F1={f1:.4f}  {verdict}")


def sweep_thresholds(prepared_images, candidate_pairs, measurements, true_pairs):
    """Перебрать сетку порогов; напечатать лучшие комбинации.
    Если у лучшей комбинации соседние значения тоже дают высокий F1 — пороги устойчивы, риск переобучения меньше.
    Замеры (measurements) уже посчитаны, поэтому каждая комбинация проверяется мгновенно."""
    results = []
    for minimum_inliers, maximum_error, minimum_detail in itertools.product(
            [20, 40, 60, 100], [0.8, 1.3, 2.0, float("inf")], [0.4, 0.5, 0.6, 0.7]):  # 4 * 4 * 4 = 64 комбинации
        # Подменяем глобальные пороги модуля main — is_duplicate() читает их при каждом вызове
        main.MINIMUM_INLIERS, main.MAXIMUM_ALIGNMENT_ERROR_PX, main.MINIMUM_DETAIL_CORRELATION = minimum_inliers, maximum_error, minimum_detail
        pairs = main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements)
        results.append((score(pairs, true_pairs)[2], minimum_inliers, maximum_error, minimum_detail))
    print("\nлучшие комбинации порогов (F1, инлайеров>=, ошибка<=, детали>=):")
    for result in sorted(results, reverse=True)[:8]:
        print(f"  F1={result[0]:.4f}  инлайеров>={result[1]:<4} ошибка<={result[2]:<4} детали>={result[3]}")


def main_check():
    true_pairs = read_pairs(TRAIN_TRUE_PAIRS_FILE)
    image_paths = main.find_input_images(TRAIN_IMAGES_FOLDER)
    print(f"обучающих картинок: {len(image_paths)}, истинных пар: {len(true_pairs)}")

    start_time = time.time()
    prepared_images, candidate_pairs, measurements = main.prepare_and_measure(image_paths)
    print(f"подготовка + проверка заняли {time.time() - start_time:.0f} с")

    # Сколько истинных пар вообще доживает до этапа кандидатов? (это потолок для recall:
    # пару, не попавшую в кандидаты, точная проверка уже не увидит)
    names = [image.name for image in prepared_images]
    candidate_name_pairs = {tuple(sorted((names[a], names[b]))) for a, b in candidate_pairs}
    print(f"истинных пар среди кандидатов: {len(true_pairs & candidate_name_pairs)}/{len(true_pairs)}\n")

    print_score("настройки main.py, без достройки групп",
                main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=False), true_pairs)
    print_score("настройки main.py, с достройкой групп",
                main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=True), true_pairs)

    if "--sweep" in sys.argv:
        sweep_thresholds(prepared_images, candidate_pairs, measurements, true_pairs)


if __name__ == "__main__":
    main_check()
