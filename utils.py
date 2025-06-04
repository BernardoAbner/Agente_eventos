# utils.py
import sys
from loguru import logger
from config import LOG_FILE_PATH, LOG_LEVEL # Certifique-se que config.py está ok

def setup_logger():
    """Configura o logger para o projeto."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        LOG_FILE_PATH,
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    return logger

# Esta é a linha crucial para exportar o app_logger:
app_logger = setup_logger()

# Para testar se este arquivo está ok, você pode adicionar temporariamente:
# if __name__ == "__main__":
#     app_logger.info("Logger do utils.py funcionando!")
# E então executar: python utils.py