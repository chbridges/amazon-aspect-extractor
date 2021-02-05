import os
import re
import unittest

import numpy as np
import pandas as pd

from src.utils.keywords import (
    extract_keywords_from_list,
    rake_str,
    keywords_to_dataframe,
    yake_str,
)

# Source of sample:
# https://www.amazon.com/gp/customer-reviews/R2LK13YHGM6HW/
# ref=cm_cr_dp_d_rvw_ttl?ie=UTF8&ASIN=B002OWETK4

sample = """I recently bought a Fjallraven bag from Urban Outfitters and
wanted to buy my friend one for her birthday. This color happened to be on
sale, and I immediately bought it. I did read the reviews, saying that some
of these bags are counterfeit, but was positive that this may have been a
mistake. I received the bag in the mail, and immediately could tell the main
difference between the bag I ordered from Urban, and the one I received from
Amazon. This bag seemed to be made of cloth, was a little more slouchy, and is
definitely less waterproof than my bag. My bag is made of a thick, waterproof
material, and compared to this one, could probably be submerged in water and be
just fine. To be honest, this was the only thing that I could tell was off. The
bag looks great, has all of the same details that mine does, and it is in a
super cute color! I do not mind that this bag may be made out of a different
material, since I did pay such a low price for it. I do not plan on returning
it, because my friend loves it! Buyers, beware that you may not be getting the
same quality bag as Fjallraven sells, but the quality is still good, and
comparable to a Jansport backpack. I would still recommend this bag, as it is
super cute, affordable on Amazon, and very trendy nowadays."""


class RakeStrTest(unittest.TestCase):
    """Tests for rake_str"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, rake_str, None)
        self.assertRaises(TypeError, rake_str, 0)
        self.assertRaises(TypeError, rake_str, [])

    def test_empty_string(self):
        self.assertEqual(len(rake_str("")), 0)
        self.assertEqual(rake_str(""), [])

    def test_sample(self):
        sample_keywords = rake_str(sample)
        # Type correctness
        self.assertGreater(len(sample_keywords), 0)
        self.assertIsInstance(sample_keywords, list)
        self.assertIsInstance(sample_keywords[0], tuple)
        self.assertIsInstance(sample_keywords[0][0], str)
        self.assertIsInstance(sample_keywords[0][1], float)
        # The following assume the NLTK stoplist, which is the default
        self.assertIn(("definitely less waterproof", 8.5), sample_keywords)
        self.assertEqual(
            sample_keywords[0], ("definitely less waterproof", 8.5)
        )  # correct sorting
        # Using the SmartStoplist, the above keyword should be missing
        self.assertGreater(len(rake_str(sample, useNLTK=False)), 0)
        self.assertNotIn(
            ("definitely less waterproof", 8.5),
            rake_str(sample, useNLTK=False),
        )
        self.assertGreater(
            len(sample_keywords),
            len(rake_str(sample, useNLTK=False)),
        )


class RakeListTest(unittest.TestCase):
    """Tests for extract_keywords_from_list"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, extract_keywords_from_list, None)
        self.assertRaises(TypeError, extract_keywords_from_list, 0)
        self.assertRaises(TypeError, extract_keywords_from_list, "")

    def test_empty_list(self):
        self.assertEqual(len(extract_keywords_from_list([])), 0)
        self.assertEqual(extract_keywords_from_list([]), [])

    def test_sample(self):
        sample_keywords_str = rake_str(sample)
        sample_keywords_list = extract_keywords_from_list([sample])

        self.assertEqual(sample_keywords_str, sample_keywords_list)
        self.assertEqual(sample_keywords_list, extract_keywords_from_list([sample, ""]))
        self.assertNotEqual(
            sample_keywords_list, extract_keywords_from_list([sample, sample])
        )
        self.assertGreater(
            len(extract_keywords_from_list([sample, sample])),
            len(sample_keywords_list),
        )
        self.assertEqual(
            set(sample_keywords_list),
            set(extract_keywords_from_list([sample, sample])),
        )


class YakeStrTest(unittest.TestCase):
    """Tests for yake_str"""

    def test_invalid_inputs(self):
        self.assertRaises(AttributeError, yake_str, None)
        self.assertRaises(AttributeError, yake_str, 0)
        self.assertRaises(TypeError, yake_str, [])

    def test_empty_string(self):
        self.assertEqual(yake_str(""), [])

    def test_sample(self):
        sample_keywords = yake_str(sample)
        # Type correctness
        self.assertGreater(len(sample_keywords), 0)
        self.assertIsInstance(sample_keywords, list)
        self.assertIsInstance(sample_keywords[0], tuple)
        self.assertIsInstance(sample_keywords[0][0], str)
        self.assertIsInstance(sample_keywords[0][1], float)
        # Sample entry
        self.assertIn(("urban outfitters", 0.016481602005070258), sample_keywords)
        self.assertEqual(
            sample_keywords[0], ("urban outfitters", 0.016481602005070258)
        )  # correct sorting


class YakeListTest(unittest.TestCase):
    """Tests for extract_keywords_from_list"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, extract_keywords_from_list, None, "yake")
        self.assertRaises(TypeError, extract_keywords_from_list, 0, "yake")
        self.assertRaises(TypeError, extract_keywords_from_list, "", "yake")

    def test_empty_list(self):
        self.assertEqual(len(extract_keywords_from_list([], "yake")), 0)
        self.assertEqual(extract_keywords_from_list([], "yake"), [])

    def test_sample(self):
        sample_keywords_str = yake_str(sample)
        sample_keywords_list = extract_keywords_from_list([sample], "yake")

        self.assertEqual(sample_keywords_str, sample_keywords_list)
        self.assertEqual(
            sample_keywords_list, extract_keywords_from_list([sample, ""], "yake")
        )
        self.assertNotEqual(
            sample_keywords_list, extract_keywords_from_list([sample, sample], "yake")
        )
        self.assertGreater(
            len(extract_keywords_from_list([sample, sample], "yake")),
            len(sample_keywords_list),
        )
        self.assertEqual(
            set(sample_keywords_list),
            set(extract_keywords_from_list([sample, sample], "yake")),
        )


class DataFrameTest(unittest.TestCase):
    """Tests for keywords_to_dataframe"""

    def test_invalid_inputs(self):
        self.assertRaises(ValueError, keywords_to_dataframe, 0)
        self.assertRaises(ValueError, keywords_to_dataframe, "")
        self.assertRaises(ValueError, keywords_to_dataframe, [""])

    def test_empty_list(self):
        empty_df = keywords_to_dataframe([])

        self.assertIsInstance(empty_df, pd.core.frame.DataFrame)
        self.assertTrue(empty_df.equals(keywords_to_dataframe(None)))
        self.assertEqual(len(empty_df), 0)
        self.assertEqual(tuple(empty_df.columns), ("keyword", "relevancy"))
        self.assertEqual(tuple(empty_df.dtypes), (object, object))

    def test_sample(self):
        sample_df = keywords_to_dataframe(rake_str(sample))

        self.assertIsInstance(sample_df, pd.core.frame.DataFrame)
        self.assertTrue(
            sample_df.equals(
                keywords_to_dataframe(extract_keywords_from_list([sample]))
            )
        )
        self.assertGreater(len(sample_df), 0)
        self.assertEqual(tuple(sample_df.dtypes), (object, np.float64))
        self.assertEqual(tuple(sample_df.columns), ("keyword", "relevancy"))
        self.assertEqual(tuple(sample_df.iloc[0]), ("definitely less waterproof", 8.5))

    def test_csv(self):
        os.chdir("src")
        sample_df = keywords_to_dataframe(rake_str(sample), csv_name="test")
        os.chdir("..")

        self.assertTrue("test.csv" in os.listdir("src/data"))
        self.assertTrue(sample_df.equals(pd.read_csv("src/data/test.csv")))
        os.remove("src/data/test.csv")
        self.assertFalse("test.csv" in os.listdir("src/data"))


if __name__ == "__main__":
    unittest.main()
