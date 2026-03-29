from test_device_utils import *
from test_functions import *
from test_deviceset import *
from test_device import *
from test_projection import *
from unittest import TestCase
import unittest
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__ + '/../../')))
sys.path.append(dirname(realpath(__file__ + '/../')))

if __name__ == "__main__":
  unittest.main()
