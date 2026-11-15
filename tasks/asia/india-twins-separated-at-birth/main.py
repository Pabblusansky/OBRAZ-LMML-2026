"""
Индия — «Близнецы, разлучённые при рождении»: найти все пары картинок, которые получены из ОДНОЙ исходной фотографии.

ЗАПУСК
    python main.py                  # ищет картинки (или .zip-архивы) в текущей папке
    python main.py path/to/input    # папка или .zip с картинками
    -> записывает output.csv (колонки: image_a,image_b) в текущую рабочую папку

ЗАДАЧА НА ОДНОЙ КАРТИНКЕ
    img_A.jpg  (612x408)  ──уменьшили, обрезали, повернули на 3°, пересохранили, осветлили──▶  img_B.jpg (544x363) = ДУБЛИКАТ
    img_C.jpg  тот же памятник, тот же ракурс, но снят на 2 секунды позже                       = НЕ дубликат (ловушка)

    Главная идея: копия (B) — это та же матрица пикселей, к которой применили ОДНО геометрическое
    преобразование (поворот + масштаб + сдвиг) и правку яркости. Поэтому её можно ТОЧНО наложить
    на оригинал. Второй снимок (C) так наложить нельзя: камера чуть сдвинулась (параллакс),
    облака/люди/листья поменялись.

КОНВЕЙЕР (каждый шаг — отдельный раздел ниже)
    1. найти входные картинки
    2. один раз подготовить каждую картинку: маленький «отпечаток» + ключевые точки
    3. КАНДИДАТЫ: для каждой картинки 10 самых похожих по отпечатку           (дёшево, ~8 с на 4000 картинок)
       -> из 4057 картинок получаем ~33 000 пар-кандидатов вместо 8,2 млн всех возможных пар
       (4057 * 4056 / 2 ≈ 8,2 млн — проверять их все точным методом слишком долго)
    4. ПРОВЕРКА каждого кандидата: сопоставить ключевые точки, подобрать одно геометрическое
       преобразование, наложить картинки, сравнить мелкие детали                 (точно, ~50 с)
    5. достроить группы: если A~B и B~C, то и A~C (по замерам вредит, по умолчанию выключено)
    6. записать output.csv

РЕЗУЛЬТАТ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ (4057 картинок, 1399 истинных пар): см. check_f1.py.

ПОЧЕМУ СООБЩЕНИЯ print() НА АНГЛИЙСКОМ
    Этот файл запускается на сервере проверяющего. Если консоль там не в UTF-8,
    print() с кириллицей упадёт с UnicodeEncodeError, и решение не засчитают.
    Латиница печатается в любой кодировке, поэтому вывод программы оставлен на английском.
"""

import csv
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ============================================================================ 0. НАСТРОЙКИ
# Пороги подобраны по F1 на обучающей выборке (check_f1.py --sweep).
# Результат был стабилен в широком диапазоне значений, поэтому выбраны значения из середины
# этого диапазона — так меньше риск переобучиться под train.

NEIGHBOURS_PER_IMAGE = 10          # сколько кандидатов брать на картинку; настоящий близнец в топ-10 у 99,9% картинок
FINGERPRINT_IMAGE_SIZE = 128       # длинная сторона (px) уменьшенной копии для отпечатков
MATCHING_IMAGE_SIZE = 512          # длинная сторона (px) копии для сопоставления ключевых точек
KEYPOINTS_PER_IMAGE = 1000         # сколько ключевых точек ORB искать на картинке
MATCH_RATIO_TEST = 0.8             # совпадение точек оставляем, только если оно заметно лучше второго по качеству (тест Лоу)
RANSAC_MAX_ERROR_PX = 4.0          # совпадение «согласно» с преобразованием, если точка попадает в радиус 4 px

MINIMUM_INLIERS = 40               # минимум 40 совпадений точек должны согласиться с одним преобразованием
MAXIMUM_ALIGNMENT_ERROR_PX = 1.3   # одно фото -> точки ложатся с ошибкой ~0,01–0,8 px; сдвинутая камера -> больше
MINIMUM_DETAIL_CORRELATION = 0.6   # после наложения мелкие детали (облака, листья, люди) должны совпадать

USE_GROUP_COMPLETION = False       # добавлять A~C, если найдены A~B и B~C. Замер на train показал, что это ВРЕДИТ
                                   # (F1 0.957 -> 0.949): одна ложная пара склеивает две группы и порождает много ложных пар.

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUT_FILE_NAME = "output.csv"


# ============================================================================ 1. ПОИСК ВХОДНЫХ КАРТИНОК
# Этот раздел не про компьютерное зрение, а про надёжность: проверяющая система несколько раз
# подсовывала «странный» вход (битый zip, tar.gz под именем .zip, указатель Git LFS вместо архива),
# поэтому код распаковывает всё, что найдёт, и при ошибке печатает понятную диагностику.

ARCHIVE_NAME_ENDINGS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".gz", ".bz2", ".xz")


def find_input_images(input_location=None):
    """
    Вернуть список путей к картинкам.
    - если передан input_location: это папка или файл-архив;
    - если ничего не передано: текущая рабочая папка.
    Каждый найденный архив (в том числе архив внутри архива) распаковывается во временную папку,
    и в ней тоже ищутся картинки.

    Пример: рабочая папка = [main.py, test-input.zip]
            -> распаковать test-input.zip в /tmp/twins_input_x/ -> найти /tmp/twins_input_x/test/img_000001.jpg, ...
    """
    start_location = Path(input_location) if input_location is not None else Path.cwd()
    folders_to_search = [start_location] if start_location.is_dir() else []
    archives_to_unpack = [start_location] if start_location.is_file() else []
    image_paths = []

    # Обход «очередью»: пока есть что распаковать или где искать — продолжаем.
    # Архивы обрабатываем первыми; распакованная папка сама попадает в список папок для поиска.
    while folders_to_search or archives_to_unpack:
        if archives_to_unpack:
            extraction_folder = extract_archive(archives_to_unpack.pop())
            if extraction_folder is not None:
                folders_to_search.append(extraction_folder)
            continue
        # os.walk с followlinks=True заходит и в папки-ссылки (Path.rglob их молча пропускает)
        for current_folder, _, file_names in os.walk(folders_to_search.pop(), followlinks=True):
            if "__MACOSX" in Path(current_folder).parts:  # мусор из zip, сделанных на macOS: файлы "._img.jpg" — не картинки
                continue
            for file_name in file_names:
                file_path = Path(current_folder) / file_name
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(file_path)
                elif file_name.lower().endswith(ARCHIVE_NAME_ENDINGS):
                    archives_to_unpack.append(file_path)

    if not image_paths:
        raise RuntimeError("no images found.\n" + describe_surroundings(start_location))
    return sorted(set(image_paths))  # set убирает повторы, sorted даёт стабильный порядок


def extract_archive(archive_path):
    """
    Распаковать архив в новую временную папку; вернуть эту папку или None, если распаковать не удалось.
    Тип архива определяется по СОДЕРЖИМОМУ файла, а не по имени: файл с именем .zip может оказаться .tar.gz.
    При ошибке печатается предупреждение, а не падение программы — чтобы можно было использовать другие входные данные.
    """
    extraction_folder = Path(tempfile.mkdtemp(prefix="twins_input_"))
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extraction_folder)
        elif tarfile.is_tarfile(archive_path):          # .tar, .tar.gz, .tar.bz2, .tar.xz (сжатие определяется автоматически)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extraction_folder)
        else:
            shutil.unpack_archive(str(archive_path), str(extraction_folder))  # последняя попытка: по расширению
        print(f"unpacked {archive_path}")
        return extraction_folder
    except Exception as error:
        print(f"WARNING: could not unpack {archive_path}: {error!r}\n{describe_file(archive_path)}", file=sys.stderr)
        return None


def describe_file(file_path):
    """Размер и первые байты файла. Первые байты («магическое число») выдают настоящий тип: b'PK' = zip, b'\\x1f\\x8b' = gzip, b'7z' = 7-Zip."""
    with open(file_path, "rb") as opened_file:
        first_bytes = opened_file.read(16)
    description = f"  {file_path}: {os.path.getsize(file_path)} bytes, starts with {first_bytes!r}"
    if first_bytes.startswith(b"version https://"):
        # Маленький текстовый файл вида "version https://git-lfs.github.com/spec/v1 / oid sha256:... / size ..." —
        # это УКАЗАТЕЛЬ Git LFS: настоящий файл так и не был скачан (на сервере нужно выполнить `git lfs pull`).
        description += ("\n    -> this is a Git LFS pointer, NOT the real archive (needs `git lfs pull`). Full content:\n"
                        + Path(file_path).read_text(errors="replace"))
    return description


def describe_surroundings(folder):
    """Перечислить содержимое входной папки (2 уровня), чтобы по тексту ошибки понять, что за вход нам дали."""
    folder = folder if folder.is_dir() else folder.parent
    lines = [f"contents of {folder}:"]
    for entry in sorted(folder.iterdir())[:50]:
        lines.append(describe_file(entry) if entry.is_file() else f"  {entry}/ ({len(list(entry.iterdir()))} entries)")
    return "\n".join(lines)


# ============================================================================ 2. ОДНОКРАТНАЯ ПОДГОТОВКА КАЖДОЙ КАРТИНКИ
# Всё, что нужно для сравнения, считаем ОДИН раз на картинку (а не на каждую пару) —
# иначе одну и ту же картинку пришлось бы обрабатывать ~16 раз (33 000 пар * 2 / 4057 картинок ≈ 16 пар на картинку).

@dataclass
class PreparedImage:
    name: str                         # имя файла, которое пишется в output.csv, например "img_000004.jpg"
    thumbnail_vector: np.ndarray      # 256 чисел: серая миниатюра 16x16 центральной части, нормированная
    colour_histogram: np.ndarray      # 128 чисел: сколько в картинке каждого оттенка/насыщенности
    matching_grayscale: np.ndarray    # серая копия, длинная сторона 512 px, для проверки наложения
    keypoint_positions: np.ndarray    # форма (число_точек, 2): координаты x, y каждой ключевой точки
    keypoint_descriptors: np.ndarray  # форма (число_точек, 32): 256-битное описание ORB каждой точки (32 байта)


def resize_to_longest_side(image, longest_side):
    """Уменьшить (никогда не увеличивать) так, чтобы max(высота, ширина) == longest_side. Для 128: 612x408 -> 128x85."""
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    if scale >= 1:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)  # INTER_AREA = усреднение пикселей -> без «лесенки»/муара


def central_part(image, kept_fraction=0.8):
    """Оставить центральные 80%: дубликаты часто слегка обрезаны или повёрнуты (чёрные углы), а центр сохраняется."""
    height, width = image.shape[:2]
    trim_y, trim_x = int(height * (1 - kept_fraction) / 2), int(width * (1 - kept_fraction) / 2)
    return image[trim_y:height - trim_y, trim_x:width - trim_x]


def make_thumbnail_vector(small_grayscale):
    """
    Миниатюра 16x16 -> 256 чисел -> вычесть среднее, поделить на длину вектора.
    После этого скалярное произведение двух векторов — это их корреляция (1 = одинаковый рисунок),
    а изменение яркости (все пиксели +20) вектор вообще не меняет (вычитание среднего его убирает).
    """
    thumbnail = cv2.resize(central_part(small_grayscale), (16, 16), interpolation=cv2.INTER_AREA)
    vector = thumbnail.astype(np.float32).flatten()
    vector -= vector.mean()
    return vector / (np.linalg.norm(vector) + 1e-6)  # 1e-6 защищает от деления на 0 у однотонной картинки


def make_colour_histogram(small_colour_image):
    """
    Посчитать пиксели по корзинам (оттенок, насыщенность): 16 корзин оттенка x 8 корзин насыщенности = 128 чисел.
    Яркость (V в HSV) специально не учитывается, поэтому у осветлённого дубликата гистограмма та же.
    Гистограмма не зависит от расположения пикселей, поэтому её не портят поворот и обрезка.
    Корень + нормировка делают скалярное произведение двух гистограмм мерой сходства от 0 до 1
    (это коэффициент Бхаттачарьи: sum(sqrt(p) * sqrt(q))).
    """
    hsv_image = cv2.cvtColor(small_colour_image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_image], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()  # оттенок в OpenCV: 0..179
    return np.sqrt(histogram / histogram.sum())


def prepare_image(image_path):
    colour_image = cv2.imread(str(image_path))  # ВНИМАНИЕ: OpenCV отдаёт каналы в порядке B, G, R
    if colour_image is None:
        return None  # нечитаемый файл -> пропускаем

    small_colour_image = resize_to_longest_side(colour_image, FINGERPRINT_IMAGE_SIZE)
    small_grayscale = cv2.cvtColor(small_colour_image, cv2.COLOR_BGR2GRAY)
    matching_grayscale = cv2.cvtColor(resize_to_longest_side(colour_image, MATCHING_IMAGE_SIZE), cv2.COLOR_BGR2GRAY)

    # ORB находит «уголковые» места (углы окон, резьба) и описывает каждое 256 битами.
    # Одна и та же физическая точка получает почти те же 256 бит даже после уменьшения / поворота / осветления.
    # ORB выбран потому, что он быстрый (бинарные описания сравниваются через XOR) и устойчив к повороту и масштабу.
    keypoints, descriptors = cv2.ORB_create(nfeatures=KEYPOINTS_PER_IMAGE).detectAndCompute(matching_grayscale, None)
    keypoint_positions = np.float32([keypoint.pt for keypoint in keypoints]) if keypoints else np.zeros((0, 2), np.float32)

    return PreparedImage(
        name=Path(image_path).name,
        thumbnail_vector=make_thumbnail_vector(small_grayscale),
        colour_histogram=make_colour_histogram(small_colour_image),
        matching_grayscale=matching_grayscale,
        keypoint_positions=keypoint_positions,
        keypoint_descriptors=descriptors,  # None, если точек не нашлось — это проверяется в measure_pair
    )


# ============================================================================ 3. ПАРЫ-КАНДИДАТЫ
# Грубый, но быстрый фильтр. Его задача — НЕ потерять настоящие пары (высокая полнота/recall),
# лишние кандидаты не страшны: их отсеет точная проверка на шаге 4.

def find_candidate_pairs(prepared_images, neighbours_per_image=NEIGHBOURS_PER_IMAGE):
    """
    similarity[i, j] = корреляция миниатюр + сходство цветовых гистограмм   (максимум 2.0)
    Для каждой картинки оставляем `neighbours_per_image` самых похожих картинок.
    Пары хранятся как (меньший индекс, больший индекс), поэтому (3, 7) и (7, 3) считаются один раз.

    Пример: топ-соседи картинки 3 — [7, 12, 5, ...] -> кандидаты (3, 7), (3, 12), (3, 5), ...
    """
    thumbnail_matrix = np.array([image.thumbnail_vector for image in prepared_images])  # (N, 256)
    histogram_matrix = np.array([image.colour_histogram for image in prepared_images])  # (N, 128)
    # Одно матричное умножение сразу даёт сходство ВСЕХ пар: элемент [i, j] = скалярное произведение векторов i и j.
    similarity = thumbnail_matrix @ thumbnail_matrix.T + histogram_matrix @ histogram_matrix.T  # (N, N)
    np.fill_diagonal(similarity, -np.inf)  # картинка не может быть кандидатом сама для себя

    neighbours_per_image = min(neighbours_per_image, len(prepared_images) - 1)
    if neighbours_per_image < 1:
        return []
    nearest_neighbours = np.argsort(-similarity, axis=1)[:, :neighbours_per_image]  # (N, k) индексы, самые похожие первыми

    # Множество (set) автоматически убирает повторы: если 3 — сосед 7, а 7 — сосед 3, пара будет одна.
    candidate_pairs = {tuple(sorted((image_index, int(neighbour_index))))
                       for image_index, neighbours in enumerate(nearest_neighbours)
                       for neighbour_index in neighbours}
    return sorted(candidate_pairs)


# ============================================================================ 4. ПРОВЕРКА ПАРЫ-КАНДИДАТА
# Точная проверка. Здесь и отличаем «копию одного фото» от «другого снимка того же памятника».

@dataclass
class PairMeasurement:
    inliers: int                     # сколько совпадений точек согласны с одним преобразованием (одно фото: обычно 100–600)
    alignment_error_px: float        # медиана расстояния между тем, куда легли согласные точки, и тем, где они должны быть
    detail_correlation: float        # корреляция мелких деталей после наложения (одно фото: обычно 0,7–0,95)


NO_MATCH = PairMeasurement(inliers=0, alignment_error_px=float("inf"), detail_correlation=0.0)  # «точно не дубликат»
keypoint_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # перебором сравнивает 256-битные описания по числу различающихся бит (расстояние Хэмминга)


def fine_details(grayscale_image):
    """Картинка минус её размытая версия = только мелкие детали (края, текстура, облака), без крупных пятен яркости.
    Это фильтр высоких частот: крупная форма здания и неба (общая для ВСЕХ фото памятника) вычитается."""
    as_float = grayscale_image.astype(np.float32)
    return as_float - cv2.GaussianBlur(as_float, (0, 0), 3)  # размытие Гаусса с сигмой 3 px


def correlation_inside_mask(first_values, second_values, mask):
    """Корреляция Пирсона значений пикселей внутри `mask`: 1 = одинаковый рисунок, 0 = никак не связаны.
    Не зависит от яркости и контраста (вычитаем среднее и делим на разброс)."""
    first = first_values[mask].astype(np.float32)
    second = second_values[mask].astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    return float((first * second).sum() / (np.sqrt((first * first).sum() * (second * second).sum()) + 1e-6))


def measure_pair(first_image, second_image):
    """
    Пошагово, с типичными числами для настоящего дубликата:
      a) сопоставить описания: 1000 vs 1000 ключевых точек -> 420 уверенных совпадений
      b) RANSAC подбирает ОДНО преобразование (масштаб + поворот + сдвиг) -> 395 из 420 совпадений согласны («инлайеры»)
      c) ошибка наложения: согласные точки ложатся в среднем в 0,02 px от своей пары
      d) переносим первую картинку на вторую, сравниваем мелкие детали -> корреляция 0,91
    Другое фото того же памятника обычно даёт: 60 инлайеров, ошибка 2,5 px, корреляция деталей 0,25.
    """
    first_descriptors, second_descriptors = first_image.keypoint_descriptors, second_image.keypoint_descriptors
    if first_descriptors is None or second_descriptors is None or len(first_descriptors) < 8 or len(second_descriptors) < 8:
        return NO_MATCH  # слишком мало точек (например, однотонная картинка) — сравнивать нечего

    # a) для каждой точки картинки 1 ищем лучшую и вторую по качеству пару в картинке 2.
    #    Оставляем, только если лучшая заметно лучше второй (тест отношения Лоу) -> убирает
    #    неоднозначные совпадения на повторяющихся узорах (одинаковые окна, арки, колонны).
    two_best_matches = keypoint_matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    confident_matches = [best for best, *others in two_best_matches
                         if others and best.distance < MATCH_RATIO_TEST * others[0].distance]
    if len(confident_matches) < 6:
        return NO_MATCH

    # queryIdx — номер точки в первой картинке, trainIdx — номер её пары во второй
    points_in_first = first_image.keypoint_positions[[match.queryIdx for match in confident_matches]]
    points_in_second = second_image.keypoint_positions[[match.trainIdx for match in confident_matches]]

    # b) RANSAC: много раз берём 2 случайных совпадения, вычисляем преобразование, которое они задают,
    #    считаем, сколько остальных совпадений согласны с ним в пределах RANSAC_MAX_ERROR_PX;
    #    оставляем преобразование с наибольшим согласием. Так ошибочные совпадения (выбросы) не портят результат.
    #    «Частичное аффинное» = поворот + одинаковый масштаб + сдвиг: ровно те правки, что делали с дубликатом
    #    (без перекоса и перспективы — они бы позволили «натянуть» и другой снимок, что дало бы ложные пары).
    transform, inlier_flags = cv2.estimateAffinePartial2D(points_in_first, points_in_second,
                                                          method=cv2.RANSAC, ransacReprojThreshold=RANSAC_MAX_ERROR_PX)
    if transform is None:
        return NO_MATCH
    inlier_flags = inlier_flags.ravel().astype(bool)
    inlier_count = int(inlier_flags.sum())

    # c) куда ложатся согласные точки? transform — матрица 2x3 = [[a, b, сдвиг_x], [c, d, сдвиг_y]]
    #    новая точка = точка @ (левая часть 2x2)^T + столбец сдвига.
    #    У копии одного фото ошибка почти 0; у снимка с другой позиции — больше (параллакс: близкие и
    #    далёкие предметы смещаются по-разному, одно преобразование не подходит всем точкам).
    #    Медиана, а не среднее — чтобы несколько неточных точек не исказили оценку.
    landed_positions = points_in_first[inlier_flags] @ transform[:, :2].T + transform[:, 2]
    alignment_error_px = float(np.median(np.linalg.norm(landed_positions - points_in_second[inlier_flags], axis=1)))

    # d) переносим картинку 1 в систему координат картинки 2; сравниваем только там, где пиксели есть у обеих
    #    (минус рамка 2 px — края после поворота «смазаны» интерполяцией)
    second_height, second_width = second_image.matching_grayscale.shape
    warped_first = cv2.warpAffine(first_image.matching_grayscale, transform, (second_width, second_height))
    # Маска перекрытия: переносим «картинку из единиц» тем же преобразованием — где получилась 1, там есть данные
    covered_area = cv2.warpAffine(np.ones_like(first_image.matching_grayscale), transform, (second_width, second_height))
    covered_area = cv2.erode(covered_area, np.ones((5, 5), np.uint8)) > 0  # эрозия ядром 5x5 срезает 2 px по краю
    if covered_area.sum() < 500:
        return PairMeasurement(inlier_count, alignment_error_px, 0.0)  # перекрытие слишком маленькое — детали не сравнить
    detail_correlation = correlation_inside_mask(fine_details(warped_first), fine_details(second_image.matching_grayscale), covered_area)

    return PairMeasurement(inlier_count, alignment_error_px, detail_correlation)


def is_duplicate(measurement):
    """Дубликат, только если выполнены ВСЕ три условия: много согласных точек, точное наложение, совпадают детали."""
    return (measurement.inliers >= MINIMUM_INLIERS
            and measurement.alignment_error_px <= MAXIMUM_ALIGNMENT_ERROR_PX
            and measurement.detail_correlation >= MINIMUM_DETAIL_CORRELATION)


# ============================================================================ 5. ДОСТРОЙКА ГРУПП
# (по умолчанию выключено, см. USE_GROUP_COMPLETION; оставлено, чтобы check_f1.py мог сравнить оба варианта)

def complete_groups(duplicate_pairs, number_of_images):
    """
    Система непересекающихся множеств (union-find): каждая картинка сначала в своей группе;
    каждая пара-дубликат объединяет две группы. Затем выводим все пары внутри каждой группы.
    Пример: найдены (1, 2) и (2, 3) -> группа {1, 2, 3} -> пары (1, 2), (1, 3), (2, 3).
    """
    group_parent = list(range(number_of_images))  # group_parent[i] — «родитель» i; корень группы — сам себе родитель

    def find_group_root(image_index):
        while group_parent[image_index] != image_index:
            group_parent[image_index] = group_parent[group_parent[image_index]]  # сокращение пути: следующие поиски быстрее
            image_index = group_parent[image_index]
        return image_index

    for first_index, second_index in duplicate_pairs:
        group_parent[find_group_root(first_index)] = find_group_root(second_index)  # объединить две группы

    members_by_group = {}
    for image_index in {index for pair in duplicate_pairs for index in pair}:
        members_by_group.setdefault(find_group_root(image_index), []).append(image_index)

    completed_pairs = set()
    for members in members_by_group.values():
        members.sort()
        completed_pairs.update((members[a], members[b]) for a in range(len(members)) for b in range(a + 1, len(members)))
    return sorted(completed_pairs)


# ============================================================================ 6. ВЕСЬ КОНВЕЙЕР + ВЫВОД

def run_parallel(function, items):
    """Применить function к каждому элементу параллельно на всех ядрах.
    Потоки (а не процессы) подходят, потому что OpenCV и numpy отпускают GIL на время тяжёлых вычислений."""
    cv2.setNumThreads(1)  # параллелим сами; запрещаем OpenCV дополнительно плодить потоки внутри каждого вызова
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return list(pool.map(function, items))


def prepare_and_measure(image_paths):
    """Шаги 2–4. Возвращаются отдельно, чтобы check_f1.py мог пробовать другие пороги без пересчёта."""
    prepared_images = [image for image in run_parallel(prepare_image, image_paths) if image is not None]
    print(f"prepared {len(prepared_images)} images")
    candidate_pairs = find_candidate_pairs(prepared_images)
    print(f"{len(candidate_pairs)} candidate pairs")
    measurements = run_parallel(lambda pair: measure_pair(prepared_images[pair[0]], prepared_images[pair[1]]), candidate_pairs)
    return prepared_images, candidate_pairs, measurements


def select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=USE_GROUP_COMPLETION):
    """Шаг 4 (решение по порогам) и шаг 5. Возвращает пары имён файлов."""
    duplicate_pairs = [pair for pair, measurement in zip(candidate_pairs, measurements) if is_duplicate(measurement)]
    if use_group_completion:
        duplicate_pairs = complete_groups(duplicate_pairs, len(prepared_images))
    name_pairs = {tuple(sorted((prepared_images[first].name, prepared_images[second].name))) for first, second in duplicate_pairs}
    return sorted(pair for pair in name_pairs if pair[0] != pair[1])  # без пар «сам с собой» (защита и от одинаковых имён файлов в разных папках)


def write_output_csv(name_pairs, output_path=OUTPUT_FILE_NAME):
    """Заголовок пишется всегда, даже если пар ноль (так требует задание)."""
    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image_a", "image_b"])
        writer.writerows(name_pairs)


def main():
    input_location = sys.argv[1] if len(sys.argv) > 1 else None
    image_paths = find_input_images(input_location)
    print(f"found {len(image_paths)} images")

    name_pairs = []
    if len(image_paths) >= 2:  # из одной картинки пару не составить
        prepared_images, candidate_pairs, measurements = prepare_and_measure(image_paths)
        name_pairs = select_duplicate_pairs(prepared_images, candidate_pairs, measurements)

    write_output_csv(name_pairs)
    print(f"wrote {len(name_pairs)} duplicate pairs to {Path(OUTPUT_FILE_NAME).resolve()}")


if __name__ == "__main__":
    main()
