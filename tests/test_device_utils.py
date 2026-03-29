from device_kit.utils import *
from unittest import TestCase
import unittest
from copy import deepcopy
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))


class TestBaseDevice(TestCase):

  def test_soc(self):
    r = np.random.random(24)
    v = soc(r, 1, 1)
    # print(v)
    # print(soc(r.reshape(24,1), 1, 1))


if __name__ == '__main__':
  unittest.main()
