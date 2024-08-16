from random import randint
from enum import Enum, auto
import logging

_LOGGER_BASE_LEVEL_ = logging.NOTSET #randint(1000, 2000)
_LOGGER_INFO_LEVEL_ = logging.INFO
_LOGGER_WARGNING_LEVEL_ = logging.WARNING
_LOGGER_DEBUG_LEVEL_ = logging.DEBUG
_LOGGER_CRITICAL_LEVEL_ = logging.CRITICAL

class Logging_Level(Enum):
    STASH = _LOGGER_BASE_LEVEL_
    DEBUG = _LOGGER_DEBUG_LEVEL_
    TEST = _LOGGER_INFO_LEVEL_
    WARNING = _LOGGER_WARGNING_LEVEL_
    INFO  = _LOGGER_CRITICAL_LEVEL_
    