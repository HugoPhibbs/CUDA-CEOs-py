import unittest
import numpy as np
import src.analysis_utils as analysis_utils

class TestRecall(unittest.TestCase):

    def test_basic(self):

        actual_indices = np.array([
            [2, 1, 8, 5, 6],
            [3, 4, 1, 2, 5],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5]
        ])

        expected_indices = np.array([
            [1, 2, 3, 4, 5],
            [3, 2, 0, 1, 4],
            [2, 1, 0, 3, 4],
            [0, 0, 1, 3, 4]
        ])

        # Expected at k = 3 is 1/4*(2/3+1/3+3/3+1/3) = 7/12

        expected_recall = 7 / 12

        k = 3

        result = analysis_utils.recall(actual_indices, expected_indices, k)

        self.assertAlmostEqual(result, expected_recall, places=5)
        
