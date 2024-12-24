import contextlib
from time import sleep
from loguru import logger

@contextlib.contextmanager
def retry(num:int,delay:int):
    for i in range(num):
        try:
            yield
        except Exception as e:
            if i == num - 1:
                raise e
            logger.warning(f"Got exception {e}, retrying in {delay} seconds")
            sleep(delay)
        