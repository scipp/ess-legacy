import unittest
import numpy as np
import scipp as sc

import operations


class OperationsTest(unittest.TestCase):
    @staticmethod
    def _run_test(val_in):
        test_input = np.array(val_in)

        # Flatten to 1D input for the moment
        test_input = sc.Variable(["y", "x"], values=test_input)
        return operations.mask_from_adj_pixels(test_input)

    def test_center_true(self):
        input_data = [[True, True, True],
                      [True, False, True],
                      [True, True, True]]

        returned = self._run_test(input_data)
        self.assertTrue(sc.all(sc.all(returned, "x"), "y").value)

    def test_center_false(self):
        input_data = [[False, False, False],
                      [False, True, False],
                      [False, False, False]]

        returned = self._run_test(input_data)
        self.assertFalse(sc.all(sc.all(returned, "x"), "y").value)

    def test_center_not_changed(self):
        input_list = [
            [[True, True, True],
             [True, False, False],
             [True, True, False]],

            [[False, False, False],
             [True, True, False],
             [True, False, False]]
        ]

        for i in input_list:
            self.assertEqual(i, self._run_test(i).values.tolist())

    def test_edges_handle_correctly(self):
        test_input = [[False, True, False],
                      [False, True, True],
                      [True, True, True]]

        # Top left should flip, all others should not
        expected = [[False, True, True],
                    [False, True, True],
                    [True, True, True]]

        self.assertEqual(expected, self._run_test(test_input).values.tolist())

    def test_5d_works(self):
        test_input = [[True, True, True, True, True],
                      [True, False, True, True, False],  # Should all -> True
                      [True, True, True, True, True],
                      [False, False, False, False, False],
                      [False, True, False, False, False]]  # Should -> False

        expected = [[True] * 5,
                    [True] * 5,
                    [True] * 5,
                    [False] * 5,
                    [False] * 5]

        self.assertEqual(expected, self._run_test(test_input).values.tolist())


if __name__ == '__main__':
    unittest.main()
