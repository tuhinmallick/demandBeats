import sys
import logging


def stdout_handler(formatter):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    return stdout_handler


def file_handler(formatter, filename='errors.log'):
    fileHandler = logging.FileHandler(filename=filename)
    fileHandler.setFormatter(formatter)
    return fileHandler


def config(handler='stdout'):
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(module)s::%(message)s')

    handlers = {'file': file_handler, 'stdout': stdout_handler}

    try:
        logHandlerfun = handlers[handler]
        logger.addHandler(logHandlerfun(formatter))
    except KeyError:
        logHandlerfun = handlers['stdout']
        logger.addHandler(logHandlerfun(formatter))

    return logger


def error(logger, *args, **kwargs):
    logger.error(*args, **kwargs)


def debug(logger, *args, **kwargs):
    logger.debug(*args, **kwargs)
