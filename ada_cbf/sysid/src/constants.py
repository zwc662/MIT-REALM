from random import randint
from enum import Enum, auto

_LOGGER_BASE_LEVEL_ = randint(1000, 2000)

class Logging_Level(Enum):
    STASH = _LOGGER_BASE_LEVEL_ + 1
    WARNING = _LOGGER_BASE_LEVEL_ + 2
    DEBUG = _LOGGER_BASE_LEVEL_ + 3
    INFO  = _LOGGER_BASE_LEVEL_ + 4
    