

import sys
import time
import itertools



spinner = itertools.cycle(['-', '/', '|', '\\'])

def write_spinner(delay=0.1):
    """
    makes a spinning wait cursor in stdout
    incorporates delay for cleanliness of interface
    """
    sys.stdout.write(spinner.next())
    sys.stdout.flush()
    time.sleep(delay)
    sys.stdout.write('\b')




