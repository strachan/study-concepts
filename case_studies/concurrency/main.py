import synchronous
import with_threads
import asynchronous
import multi_processing
import sys
import argparse

sys.path.append('..\\..')
from utils import time_func


sites = [
    "http://www.jython.org",
    "http://olympus.realpython.org/dice"
] * 80

concurrency_types = {'sync': synchronous.download_all_site}


@time_func
def start(concurrency_type):
    return concurrency_types[concurrency_type](sites)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--concurrency',
                        choices=['sync'],
                        help='choose concurrency type')
    args = vars(parser.parse_args())

    start(args['concurrency'])
