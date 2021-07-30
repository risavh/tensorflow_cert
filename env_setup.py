# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import PIL

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("_"*50)
    print('Pyton version {}'.format(sys.version))
    print("Tensorflow version {}".format(tf.__version__))
    print("TFDS version {}".format(tfds.__version__))
    print("Numpy version {}".format(np.__version__))
    print("Pandas version {}".format(pd.__version__))
    print("PIL version {}".format(PIL.__version__))
    print("_" * 50)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
