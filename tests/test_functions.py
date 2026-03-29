from device_kit.functions import *
from unittest import TestCase
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%0.6f' % (v,),
  },
)


class TestQuadraticCost2Function(TestCase):
  def test_basics(self):
    f = HLQuadraticCost(-1, 0, 0, 1)  # Turns out b = -1, a = 1/2.
    self.assertEqual(f(1), 0)
    self.assertEqual(f(0), 0.5)
    self.assertEqual(f.deriv(0), -1)
    self.assertEqual(f.deriv(1), 0.)
    self.assertEqual(f.hess([0]), f.hess(0), [[1.]])
    self.assertTrue((f.hess([1, 1]) == f.hess([253, 2])).all())


class TestSumFunction(TestCase):
  def test_basics(self):
    f = SumFunction([NullFunction(), NullFunction(), NullFunction()])
    g = HLQuadraticCost(-1, 0, 0, 1)
    self.assertEqual(f(1), 0)
    self.assertEqual(f.deriv(0), 0)
    self.assertEqual(f.hess(0), 0)
    f = SumFunction([g, g, g])
    self.assertEqual(f(1), 0)
    self.assertEqual(f(0), 1.5)
    self.assertEqual(f.deriv(0), -3)
    self.assertEqual(f.hess(0), 3)


class TestPoly2D(TestCase):
  def test_basics(self):
    f = Poly2D([[0, 0, 0], [1, 1, 1], [1, 2, 3]])
    xy = [
      [[0, 0, 0], [0, 1, 3], [0, 1, 2]],
      [[1, 1, 1], [0, 3, 6], [0, 3, 4]],
      [[2, 2, 2], [0, 7, 11], [0, 5, 6]]
    ]
    for [x, y, dy] in xy:
      self.assertTrue((y == f.vector(x)).all())
      self.assertTrue((dy == f.deriv(x)).all())
      self.assertTrue(np.array(y).sum() == f(x))
      self.assertTrue((np.diag(f.hess(x)) == [0, 2, 2]).all())


class TestPoly2DOffset(TestCase):
  def test_basics(self):
    f = Poly2DOffset([[0, 0, 0, 1], [1, 1, 1, 1], [1, 2, 3, 1]])
    xy = [
      [[-1, -1, -1], [0, 1, 3]],
      [[0, 0, 0], [0, 3, 6]],
      [[1, 1, 1], [0, 4+2+1, 4+4+3]]
    ]
    for [x, y] in xy:
      self.assertTrue((y == f.vector(x)).all())
      self.assertTrue(np.array(y).sum() == f(x))


class TestX2D(TestCase):
  def test_basics(self):
    g = HLQuadraticCost(-1, 0, 0, 1)
    h = HLQuadraticCost(0, 1, 0, 1)
    f = X2D([g, h, g])
    xy = [
      [[0, 0, 0], [0.5, 0, 0.5]],
      [[1, 1, 1], [0, 0.5, 0]],
      [[2, 2, 2], [0.5, 2, 0.5]],
      [[3, 4, 5], [2, 8, 8]]
    ]
    for [x, y] in xy:
      self.assertTrue(np.array(y).sum() == f(x))
      self.assertEqual(f.deriv(x).shape, (3,))
      self.assertEqual(f.hess(x).shape, (3, 3))
      # print(x, f.hess(x))


class TestDemandFunction(TestCase):
  def test_test(self):
    f = DemandFunction(np.poly1d([1, 1, 1]))
    x = np.ones(10)
    x[5] = 2
    self.assertEqual(f(x), 7)
    self.assertEqual(f.deriv(x)[5], 5)
    self.assertEqual(f.deriv(x)[0], f.deriv(x)[9], 0)
    self.assertEqual(f.hess(x)[5][5], 2)
    # f = DemandFunction(np.poly1d([1,0]))
    # print(f(x))
    # print(f.deriv(x))
    # print(f.hess(x))


class TestRangesFunction(TestCase):
  def test_test(self):
    a = Poly2D([[0, 0, 0], [0, 0, 0]])
    b = Poly2D([[1, 1, 1], [1, 1, 1]])
    c = Poly2D([[2, 2, 2], [2, 2, 2]])
    f = RangesFunction([((0, 2), a), ((2, 4), b), ((4, 6), c)])
    x = np.ones(6)
    # print(f(x))
    # print(f.deriv(x), a.deriv(x[0:2]), b.deriv(x[2:4]), c.deriv(x[4:6]))
    # print(f.hess(x), b.hess(x[2:4]))


if __name__ == '__main__':
  unittest.main()
