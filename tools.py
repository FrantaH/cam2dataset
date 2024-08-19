import time


# ANSI escape codes for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

ORIGINAL = 0
IMPURITY = 1
BASE = 2
RENTGEN = 3
DMC = 4
PICTOGRAMS = 5
OCR = 6

RESULTS_SIZE = 7


def time_measure(fce):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fce(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Processing time for function:", fce.__name__, "was:", elapsed_time, "seconds")
        return result
    return wrapper
