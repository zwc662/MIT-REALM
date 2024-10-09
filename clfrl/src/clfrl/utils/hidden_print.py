import os
import sys


class HiddenPrints:
    def __enter__(self):
        self.devnull = open("/dev/null", "w")
        self.oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(self.devnull.fileno(), 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.oldstdout_fno, 1)
        self.devnull.close()
