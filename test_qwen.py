"""
Qwen3-TTS (MLX 8-bit) 连续生成耗时测试脚本。

运行方式（项目根目录）:  uv run test_qwen.py

需要 Apple Silicon + mlx 依赖:  uv sync --extra mlx

下方 2 选 1，改 TTS_PRESET 即可：
  1 = 1.7B 8-bit MLX（质量更好）
  2 = 0.6B 8-bit MLX（更快）
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 二选一：1=1.7B 8-bit MLX（与参考项目相同）, 2=0.6B 8-bit MLX（更快，需单独下载）
TTS_PRESET = 1

try:
    from open_llm_vtuber.tts.qwen3_mlx_tts_adapter import TTSEngine
except ModuleNotFoundError as e:
    if e.name in ("loguru", "mlx_audio"):
        print(
            "This script must run with the project environment (dependencies not found).\n"
            "Run from the project root:  uv run test_qwen.py\n"
            "For MLX TTS install:  uv sync --extra mlx",
            file=sys.stderr,
        )
        sys.exit(1)
    raise


async def test_mac_tts():
    print("=" * 60)
    print("  Qwen3-TTS MLX 8-bit 连续生成 · 耗时测试（冷启动 vs 热启动）")
    print("=" * 60)

    # --- 阶段1：加载模型 ---
    if TTS_PRESET == 1:
        model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    else:
        model_id = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    print(f"\n⏳ [阶段1] 正在加载模型到内存（model_id={model_id}）...")
    start_load = time.perf_counter()
    tts = TTSEngine(
        model_id=model_id,
        speaker="Vivian",
        language="Auto",
        instruct="用开心、得意的语气说",
        speed=1.0,
        run_in_subprocess=False,
    )
    load_sec = time.perf_counter() - start_load
    print(f"✅ 模型加载完成！耗时: {load_sec:.2f} 秒\n")

    # --- 阶段2：首次生成（冷启动，含 MPS 图编译）---
    print("🔥 [阶段2] 首次生成（含 MPS 图编译，会较慢，请耐心等待）...")
    text_1 = "你好呀！我是测试用的虚拟主播，这是我今天说的第一句话！"
    start_first = time.perf_counter()
    out_1 = await tts.async_generate_audio(text_1, "mac_test_1")
    first_sec = time.perf_counter() - start_first
    print(f"✅ 首次生成完成！耗时: {first_sec:.2f} 秒（冷启动）")
    if out_1:
        print(f"   输出: {out_1}\n")

    # --- 阶段3：第二次生成（热启动）---
    print("⚡️ [阶段3] 第二次生成（热启动，体现实际直播时的速度）...")
    text_2 = (
        "你看！当我再说第二句话的时候，是不是瞬间就生成出来了？这才是 M2 的真正实力！"
    )
    start_second = time.perf_counter()
    out_2 = await tts.async_generate_audio(text_2, "mac_test_2")
    second_sec = time.perf_counter() - start_second
    print(f"🚀 第二次生成完成！耗时: {second_sec:.2f} 秒（热启动）")
    if out_2:
        print(f"   输出: {out_2}\n")

    # --- 阶段4：再连续一句（巩固热启动观感）---
    print("⚡️ [阶段4] 第三句（继续热启动）...")
    text_3 = "所以呀，第一次慢是正常的，后面就会越来越快，这就是 GPU 图编译的魔法！"
    start_third = time.perf_counter()
    out_3 = await tts.async_generate_audio(text_3, "mac_test_3")
    third_sec = time.perf_counter() - start_third
    print(f"🚀 第三句完成！耗时: {third_sec:.2f} 秒\n")

    # --- 汇总 ---
    print("=" * 60)
    print("  耗时汇总")
    print("=" * 60)
    print(f"  模型加载:     {load_sec:.2f} s")
    print(f"  首次生成:     {first_sec:.2f} s  (冷启动)")
    print(f"  第二次生成:   {second_sec:.2f} s  (热启动)")
    print(f"  第三句生成:   {third_sec:.2f} s  (热启动)")
    print("=" * 60)

    # 可选：用系统播放器播最后一段
    if out_3 and os.path.exists(out_3):
        print("\n🎵 正在播放第三句生成的音频...")
        os.system(f"afplay '{out_3}'")

    _print_explanation()


def _print_explanation():
    print("\n" + "=" * 60)
    print("  代码说明（为何这样写）")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mac_tts())
