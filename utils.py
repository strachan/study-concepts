import numpy as np
import time


def time_func(f):
    def wrap(*args):
        repetitions = 100
        time_result = np.zeros(repetitions)
        for i in np.arange(0, repetitions):
            time1 = time.time()
            ret = f(*args)
            time2 = time.time()
            time_result[i] = (time2 - time1) * 1000.0
        print(f'{f.__name__} function took {np.mean(time_result)} ms')
        return ret
    return wrap
