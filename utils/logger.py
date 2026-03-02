"""统一的日志配置"""
import logging
import os

# 从环境变量获取日志级别，默认为 INFO
LOG_LEVEL = os.environ.get('QUANT_LOG_LEVEL', 'INFO').upper()


def get_logger(name: str) -> logging.Logger:
    """获取配置好的 logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
