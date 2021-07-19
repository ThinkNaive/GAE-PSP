import sys
from loguru import logger


def init_logger(output_dir, level="DEBUG"):
    logfile = "{}/training.log".format(output_dir)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="<green>[{time:YY-MM-DD HH:mm:ss}]</green> <level>{level:<8}</level> <lc>{file}:{function}:{line}</lc> - <level>{message}</level>",
    )
    logger.add(
        logfile,
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="<green>[{time:YY-MM-DD HH:mm:ss}]</green> <level>{level:<8}</level> <lc>{file}:{function}:{line}</lc> - <level>{message}</level>",
    )
