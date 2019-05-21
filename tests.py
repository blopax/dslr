import pandas as pd

import unittest
import describe_utils
import binary_classifier


class TestDslr(unittest.TestCase):
    def test_get_quantile(self):
        s = pd.Series([1, 2, 3])
        self.assertEqual(describe_utils.get_quantile(s, 3, 1, 2), 2)

        s = pd.Series([-5, 0, 3.5, 5.4, 6.2, 58])
        self.assertEqual(describe_utils.get_quantile(s, 6, 1, 2), 4.45)
        self.assertEqual(describe_utils.get_quantile(s, 6, 1, 4), 0.875)
        self.assertEqual(describe_utils.get_quantile(s, 6, 3, 4), 6.0)

        s = pd.Series([x for x in range(101)])
        self.assertEqual(describe_utils.get_quantile(s, 101, 99, 100), 99)

        s = pd.Series([])
        self.assertEqual(describe_utils.get_quantile(s, 100, 25, 100), None)

    def test_get_count_mean(self):
        s = pd.Series([1, 2, 3])
        self.assertEqual(describe_utils.get_count_mean(s), (3, 2))

        s = pd.Series([-5, 0, 3.5, 5.4, 6.2, 58])
        self.assertEqual(describe_utils.get_count_mean(s), (6, 11.35))

        s = pd.Series([x for x in range(101)])
        self.assertEqual(describe_utils.get_count_mean(s), (101, 50))

        s = pd.Series([])
        self.assertEqual(describe_utils.get_count_mean(s), (0, None))

    def test_get_std(self):
        s = pd.Series([1, 2, 3])
        self.assertEqual(describe_utils.get_std(s, 3, 2), 1)

        s = pd.Series([-5, 0, 3.5, 5.4, 6.2, 58])
        self.assertAlmostEqual(describe_utils.get_std(s, 6, 11.35), 23.22, 2)

        s = pd.Series([x for x in range(101)])
        self.assertAlmostEqual(describe_utils.get_std(s, 101, 50), 29.3, 2)

        s = pd.Series([])
        self.assertEqual(describe_utils.get_std(s, 0, None), None)

    def test_hypothesis(self):
        x = pd.DataFrame([[0, 0, 0]])
        thetas = [0, 0, 0]
        self.assertAlmostEqual(binary_classifier.hypothesis(x, thetas), 0.5, 2)

        x = pd.DataFrame([[1, 2, 5]])
        thetas = [1, 2, -1]
        self.assertAlmostEqual(binary_classifier.hypothesis(x, thetas), 0.5, 2)

        x = pd.DataFrame([0])
        thetas = [5]
        self.assertAlmostEqual(binary_classifier.hypothesis(x, thetas), 0.5, 2)

        x = pd.DataFrame([[1000, 10, 20, 15]])
        thetas = [1000, 10, 20, -23]
        self.assertAlmostEqual(binary_classifier.hypothesis(x, thetas), 1, 2)


if __name__ == "__main__":
    unittest.main()
