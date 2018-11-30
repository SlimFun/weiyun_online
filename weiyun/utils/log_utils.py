import logging
import sys

logger = logging.getLogger()    # initialize logging class
logger.setLevel(logging.INFO)  # default log level
format = logging.Formatter("%(message)s")    # output format
sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

