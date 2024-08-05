from random import randint
from enum import Enum, auto

_LOGGER_BASE_LEVEL_ = randint(1000, 2000)

class Logging_Level(Enum):
    
    STASH = _LOGGER_BASE_LEVEL_ + 1
    DEBUG = STASH + 1
    TEST = DEBUG + 1
    WARNING = TEST + 1
    INFO  = WARNING + 1
    