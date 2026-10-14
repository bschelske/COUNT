"""
Tests for the filename helpers in COUNT/bg_plotting.py.

Run from the repository root:
    python -m unittest discover COUNT/tests
"""
import unittest

from COUNT.bg_plotting import get_frequency, is_trial_results_file


class TestGetFrequency(unittest.TestCase):
    def test_filenames_that_already_worked(self):
        examples = {
            'test 120_khz': 120,
            '95kHz8ampHICONC019_results': 95,
            '25 kHz_7.2Vp-p': 25,
            'A375_7.2Vpp_10kHz001': 10,
            'A375 200kHz 5.8Vp-p_proof': 200,
            '23Feb2024 Non RosetteSep 20kHz001': 20,
            'a375 120 khz_results.csv': 120,
        }
        for name, frequency in examples.items():
            self.assertEqual(get_frequency(name), frequency, name)

    def test_number_in_cell_name(self):
        self.assertEqual(get_frequency('exotic_cell_type2 15khz_results.csv'), 15)  # used to give 215

    def test_four_digit_and_decimal_frequencies(self):
        self.assertEqual(get_frequency('A375_1000kHz_results.csv'), 1000)  # used to give "000"
        self.assertEqual(get_frequency('A375_12.5kHz_results.csv'), 12.5)  # used to give 5

    def test_no_frequency(self):
        self.assertIsNone(get_frequency('A375_results.csv'))


class TestTrialResultsFile(unittest.TestCase):
    def test_trial_results_files(self):
        self.assertTrue(is_trial_results_file('results/A375 10kHz_results.csv'))
        self.assertFalse(is_trial_results_file('results/A375 10kHz_trajectory_results.csv'))
        self.assertFalse(is_trial_results_file('results/final_results/final_results.csv'))
        self.assertFalse(is_trial_results_file('results/notes.csv'))


if __name__ == '__main__':
    unittest.main()
