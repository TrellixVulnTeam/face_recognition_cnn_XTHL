#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri June 22 16:15:10 2022
test_prediction.py python file
@author: Im-Rises
"""

import unittest


class MyTestCase(unittest.TestCase):
    """
    Assert on error class unit test
    """

    def test_something(self):
        """
        Assert on error function
        :return:
        """
        self.assertEqual(True, True)  # add assertion here


if __name__ == "__main__":
    unittest.main()
