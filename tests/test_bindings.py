import unittest
import numpy as np
import cuda_ceos_py as ceos


class coCEOsTest(unittest.TestCase):

    def test_no_errors(self):
        n = 10_000
        d = 30
        D = 12
        m = 50
        
        X = np.random.normal(size=(n, d)).astype(np.float32)

        try:
            L, S, R = ceos.indexing_coCEOs(X, D, m)
        except:
            self.fail("coCEOs indexing raised an exception unexpectedly!")


        