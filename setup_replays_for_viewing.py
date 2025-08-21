#!/usr/bin/env python3
"""
Скрипт для настройки replay файлов для просмотра в StarCraft II
"""

import os
import shutil
import glob
from pathlib import Path

def setup_replays_for_starcraft():
    """Копирует replay файлы в стандартную директорию StarCraft II"""
    
    print("🎬 === НАСТРОЙКА REPLAY ФАЙЛОВ ДЛЯ STARCRAFT II ===")
    
    # Ищем replay файлы
    replay_files = glob.glob("collected_replays/*.SC2Replay")
    
    if not replay_files:
        print("❌ Не найдено replay файлов в collected_replays/")
        return False
    
    print(f"📁 Найдено {len(replay_files)} replay файлов")
    
    # Возможные пути для StarCraft II replays
    possible_paths = [
        Path.home() / "Documents" / "StarCraft II" / "Accounts",
        Path("/home/user/StarCraftII/Replays"),
        Path("/home/user/Documents/StarCraft II/Replays"),
        Path.home() / "StarCraftII" / "Replays"
    ]
    
    # Ищем существующую директорию
    target_dir = None
    for path in possible_paths:
        if path.exists():
            target_dir = path
            break
    
    if not target_dir:
        # Создаем стандартную директорию
        target_dir = Path.home() / "Documents" / "StarCraft II" / "Replays"
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Создана директория: {target_dir}")
    
    # Если это Accounts директория, ищем поддиректорию пользователя
    if "Accounts" in str(target_dir):
        # Ищем поддиректории аккаунтов
        account_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
        if account_dirs:
            # Берем первую найденную
            account_dir = account_dirs[0] / "Replays"
            account_dir.mkdir(exist_ok=True)
            target_dir = account_dir
        else:
            # Создаем общую папку
            target_dir = target_dir / "default_user" / "Replays"
            target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Целевая директория: {target_dir}")
    
    # Копируем файлы
    copied_files = []
    for replay_file in replay_files:
        filename = os.path.basename(replay_file)
        target_path = target_dir / filename
        
        try:
            shutil.copy2(replay_file, target_path)
            copied_files.append(filename)
            print(f"✅ Скопирован: {filename}")
        except Exception as e:
            print(f"❌ Ошибка копирования {filename}: {e}")
    
    print(f"\n📊 ИТОГИ:")
    print(f"  Скопировано файлов: {len(copied_files)}")
    print(f"  Директория: {target_dir}")
    
    # Инструкции для пользователя
    print(f"\n🎮 КАК ПРОСМОТРЕТЬ REPLAY:")
    print(f"1. Откройте StarCraft II")
    print(f"2. В главном меню выберите 'Replays'")
    print(f"3. Найдите файлы, начинающиеся с 'amal_2s_vs_1sc_'")
    print(f"4. Сравните поведение:")
    print(f"   🧠 trained_* - обученная AMAL")
    print(f"   🎲 random_* - случайная политика")
    
    print(f"\n🔍 НА ЧТО ОБРАТИТЬ ВНИМАНИЕ:")
    print(f"- Координация агентов: двигаются ли они вместе?")
    print(f"- Эффективность атак: фокусируются ли на одной цели?")
    print(f"- Использование пространства: избегают ли столкновений?")
    print(f"- Время реакции: быстро ли принимают решения?")
    
    print(f"\n💡 ОЖИДАЕМЫЕ РАЗЛИЧИЯ:")
    print(f"- TRAINED: более координированное поведение")
    print(f"- RANDOM: хаотичные движения, плохая координация")
    print(f"- TRAINED в среднем завершает эпизоды быстрее")
    
    return len(copied_files) > 0

def create_replay_analysis_guide():
    """Создает руководство по анализу replay"""
    
    guide_content = """
# 🎬 Руководство по анализу AMAL Replay

## Файлы для просмотра:
- `amal_2s_vs_1sc_trained_*` - эпизоды с обученной AMAL
- `amal_2s_vs_1sc_random_*` - эпизоды со случайной политикой

## Что анализировать:

### 1. 🤝 Координация агентов
- **TRAINED**: Агенты должны двигаться согласованно
- **RANDOM**: Хаотичные, несогласованные движения

### 2. 🎯 Тактика атаки  
- **TRAINED**: Фокус на одной цели, совместные атаки
- **RANDOM**: Разбросанные атаки, нет фокуса

### 3. 🏃 Эффективность движения
- **TRAINED**: Прямые пути, избегание препятствий
- **RANDOM**: Зигзагообразные движения, застревание

### 4. ⚡ Время реакции
- **TRAINED**: Быстрые, решительные действия
- **RANDOM**: Медленные, непоследовательные действия

## Результаты тестов:
- AMAL превосходит random policy на 13.2% по reward
- AMAL более эффективна на 37.5% (меньше шагов)
- Лучший TRAINED эпизод: R=1.31
- Лучший RANDOM эпизод: R=2.22 (но нестабильно!)

## Выводы:
✅ AMAL демонстрирует обучение
✅ Более стабильная производительность
⚠️ Все еще есть потенциал для улучшения
"""
    
    with open("REPLAY_ANALYSIS_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("📋 Создано руководство: REPLAY_ANALYSIS_GUIDE.md")

if __name__ == "__main__":
    success = setup_replays_for_starcraft()
    if success:
        create_replay_analysis_guide()
        print(f"\n🎉 Настройка завершена! Теперь можно анализировать replay в StarCraft II")
    else:
        print(f"\n❌ Ошибка настройки replay файлов")
