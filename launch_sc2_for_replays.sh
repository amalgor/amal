#!/bin/bash
# 🎮 Скрипт для запуска StarCraft II для просмотра replay

echo "🎬 === ЗАПУСК STARCRAFT II ДЛЯ ПРОСМОТРА REPLAY ==="

# Путь к SC2
SC2_PATH="/home/user/StarCraftII/Versions/Base75689/SC2_x64"
REPLAY_DIR="/home/user/StarCraftII/Replays"

echo "📋 Проверяем наличие компонентов..."

# Проверяем SC2
if [ ! -f "$SC2_PATH" ]; then
    echo "❌ StarCraft II не найден по пути: $SC2_PATH"
    echo "💡 Попробуйте найти SC2_x64 в других местах или установить SC2"
    exit 1
else
    echo "✅ StarCraft II найден: $SC2_PATH"
fi

# Проверяем replay файлы
if [ ! -d "$REPLAY_DIR" ]; then
    echo "❌ Директория replay не найдена: $REPLAY_DIR"
    exit 1
fi

REPLAY_COUNT=$(ls "$REPLAY_DIR"/*.SC2Replay 2>/dev/null | wc -l)
if [ $REPLAY_COUNT -eq 0 ]; then
    echo "❌ Replay файлы не найдены в $REPLAY_DIR"
    echo "💡 Запустите сначала: python setup_replays_for_viewing.py"
    exit 1
else
    echo "✅ Найдено $REPLAY_COUNT replay файлов"
fi

echo ""
echo "🎯 ВАШИ REPLAY ФАЙЛЫ:"
ls -la "$REPLAY_DIR"/*.SC2Replay | while read line; do
    filename=$(basename "$(echo "$line" | awk '{print $9}')")
    size=$(echo "$line" | awk '{print $5}')
    
    if [[ $filename == *"trained"* ]]; then
        echo "  🧠 $filename ($size bytes)"
    elif [[ $filename == *"random"* ]]; then
        echo "  🎲 $filename ($size bytes)"
    else
        echo "  📄 $filename ($size bytes)"
    fi
done

echo ""
echo "🚀 Запускаем StarCraft II..."
echo "   После запуска:"
echo "   1. Перейдите в меню REPLAYS"
echo "   2. Найдите файлы amal_2s_vs_1sc_*"
echo "   3. Выберите и нажмите Watch Replay"

# Запускаем SC2
"$SC2_PATH" &

echo ""
echo "✅ StarCraft II запущен в фоновом режиме"
echo "🎬 Приятного просмотра ваших AI агентов!"
echo ""
echo "💡 ПОДСКАЗКИ:"
echo "   - Начните с файла: trained_0_*_r1.31_s92 (лучший результат)"
echo "   - Затем сравните с: random_0_*_r0.63_s152"
echo "   - Используйте +/- для изменения скорости"
echo "   - TAB для переключения камеры"
