"""vLLM 日志配置生成工具。

该脚本根据环境变量自动生成 vLLM 日志配置文件（JSON格式），
并输出文件路径，供 VLLM_LOGGING_CONFIG_PATH 环境变量使用。

主要功能：
  • 根据环境变量（VLLM_LOGGING_LEVEL, VLLM_LOGGING_COLOR 等）动态生成配置

  • 使用 ConcurrentRotatingFileHandler 实现日志文件绕接（多进程安全）

  • 清理旧的日志文件，避免磁盘空间泄漏


用法:
    export VLLM_LOGGING_CONFIG_PATH=$(python update_logging_config.py --logging-file-name /path/to/server.log)
"""
import argparse
import json
import os
import sys
import tempfile
import glob
from typing import Dict, Any, Optional

# 从环境变量或默认值获取配置
# 这些变量在模块导入时读取，后续修改环境变量不会生效
VLLM_LOGGING_PREFIX = os.getenv("VLLM_LOGGING_PREFIX", "")
VLLM_LOGGING_LEVEL = os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper()
VLLM_LOGGING_COLOR = os.getenv("VLLM_LOGGING_COLOR", "auto")
VLLM_LOGGING_STREAM = os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout")

_FORMAT = f"{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s (%(fileinfo)s:%(lineno)d) %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _use_color() -> bool:
    """判断是否应该使用彩色输出。

    判断优先级：
    1. NO_COLOR 非空 → 关闭颜色
    2. VLLM_LOGGING_COLOR="0" → 关闭颜色
    3. VLLM_LOGGING_COLOR="1" → 启用颜色
    4. 如果输出流指向终端（tty）→ 启用颜色
    5. 其他情况 → 关闭颜色

    Returns:
        bool: True 使用彩色formatter，False 使用普通formatter
    """
    no_color = os.getenv("NO_COLOR", "").strip()
    if no_color or VLLM_LOGGING_COLOR == "0":
        return False
    if VLLM_LOGGING_COLOR == "1":
        return True
    if VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


def _build_logging_config(log_file_name: str) -> Dict[str, Any]:
    """构建完整的日志配置字典。

    Args:
        log_file_name: 日志文件路径（支持绝对路径或相对路径）

    Returns:
        dict: vLLM logging.config.dictConfig 兼容的配置字典

    配置包含：
        ▪ 文件handler: ConcurrentRotatingFileHandler，单文件最大100MB，保留30个备份

        ▪ 控制台handler: StreamHandler，根据 _use_color() 选择彩色/普通formatter

        ▪ logger: 仅记录 vllm 命名空间下的日志，propagate=False

    """
    formatter_name = "vllm_color" if _use_color() else "vllm"

    return {
        "omni_logging_format": True,
        "formatters": {
            "vllm": {
                "class": "vllm.logging_utils.NewLineFormatter",
                "datefmt": _DATE_FORMAT,
                "format": _FORMAT,
            },
            "vllm_color": {
                "class": "vllm.logging_utils.ColoredFormatter",
                "datefmt": _DATE_FORMAT,
                "format": _FORMAT,
            },
        },
        "handlers": {
            "vllm": {
                "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
                "formatter": "vllm",
                "level": VLLM_LOGGING_LEVEL,
                "maxBytes": 104857600,
                "backupCount": 30,
                "filename": log_file_name,
                "delay": True,
                "use_gzip": False,
                "encoding": "utf-8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "level": VLLM_LOGGING_LEVEL,
                "stream": VLLM_LOGGING_STREAM,
            },
        },
        "loggers": {
            "vllm": {
                "handlers": ["vllm", "console"],
                "level": VLLM_LOGGING_LEVEL,
                "propagate": False,
            },
        },
        "version": 1,
        "disable_existing_loggers": False,
    }


def delete_old_log_files(log_file_path: str) -> None:
    """删除旧的日志文件及已绕接的备份文件。

    会删除目标日志文件本身，以及匹配 "{log_file_path}.*" 的所有备份文件
    （如 app.log.1, app.log.2 等）。

    Args:
        log_file_path: 要删除的主日志文件路径
    """
    try:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        rotated_pattern = f"{log_file_path}.*"
        for rotated_file in glob.glob(rotated_pattern):
            os.remove(rotated_file)
    except Exception as e:
        sys.stderr.write(f"警告: 删除日志文件时出错: {e}\n")


def save_config(config: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """将配置字典保存为 JSON 文件。

    Args:
        config: 要保存的配置字典
        output_path: 输出文件路径。若不指定，则创建临时文件

    Returns:
        str: 生成文件的绝对路径

    Raises:
        SystemExit: 文件创建或写入失败时退出
    """
    if not output_path:
        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', prefix='logging_config_', delete=False, encoding='utf-8'
            )
            output_path = temp_file.name
            temp_file.close()
        except Exception as e:
            sys.exit(f"错误: 创建临时文件失败: {e}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return output_path
    except Exception as e:
        sys.exit(f"错误: 写入文件失败: {e}")


def main():
    """CLI 入口函数。

    解析命令行参数，依次执行：清理旧日志 → 生成配置 → 保存文件

    CLI 参数:
        --logging-file-name: [必需] 日志文件路径
        --output: [可选] 输出配置文件路径（不指定则创建临时文件）
        --no-delete-old: [可选] 不删除旧日志文件

    Returns:
        int: 退出码，0表示成功
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-file-name", required=True)
    parser.add_argument("--output", help="输出文件路径（如果不指定，将创建临时文件）")
    parser.add_argument("--no-delete-old", action="store_true", help="不删除旧的日志文件")
    args = parser.parse_args()

    if not args.no_delete_old:
        delete_old_log_files(args.logging_file_name)

    config = _build_logging_config(args.logging_file_name)

    output_path = save_config(config, args.output)

    sys.stdout.write(output_path)
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    # 用法: export VLLM_LOGGING_CONFIG_PATH=$(python update_logging_config.py --logging-file-name ${LOG_PATH}/server_0.log)
    sys.exit(main())
