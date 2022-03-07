import os
import time


def timed(function):
    def timed_function(*args, **kwargs):
        start_time = time.perf_counter()
        result = function(*args, *kwargs)
        elapsed_time = time.perf_counter() - start_time
        print("Time to Complete Function: " + str(round(elapsed_time, 2)))
        return result
    return timed_function

@timed
def inefficient_function(directory, max_files=100):
    """ return the sum of the lengths of all lines, in all files in directory """
    list_files = os.listdir(directory)
    sum = 0
    for i in range(min(len(list_files), max_files)):
        filepath = os.path.join(directory, list_files[i])
        with open(filepath, "r") as file:
            contents = file.read()
        for line in contents.split("\n"):
            sum += len(line)
    return sum
