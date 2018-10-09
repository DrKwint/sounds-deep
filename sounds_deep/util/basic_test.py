import sounds_deep.util.basic as basic
import unittest
import tensorflow as tf


class TestRunEpochOpsMethods(unittest.TestCase):
    def setUp(self):
        self.session = tf.Session()
        self.random_sum = tf.add(
            tf.random_normal([2, 3]), tf.random_normal([3]))

    def test_output_shape(self):
        """
        Verify that the first element of the output shape is equal to the 
        number of steps taken
        """
        steps = 3  # any integer
        output = basic.run_epoch_ops(
            self.session, steps, verbose_ops_dict={'sum': self.random_sum})
        self.assertEqual(output['sum'].shape[0], steps)

    def test_empty_input(self):
        """Ensure that leaving out optional parameters won't break anything"""
        basic.run_epoch_ops(self.session, 5)


if __name__ == '__main__':
    unittest.main()
