import unittest
from src.utils.keywords import extract_keyword_str, extract_keyword_list

sample = "I recently bought a Fjallraven bag from Urban Outfitters and wanted to buy my friend one for her birthday. This color happened to be on sale, and I immediately bought it. I did read the reviews, saying that some of these bags are counterfeit, but was positive that this may have been a mistake. I received the bag in the mail, and immediately could tell the main difference between the bag I ordered from Urban, and the one I received from Amazon. This bag seemed to be made of cloth, was a little more slouchy, and is definitely less waterproof than my bag. My bag is made of a thick, waterproof material, and compared to this one, could probably be submerged in water and be just fine. To be honest, this was the only thing that I could tell was off. The bag looks great, has all of the same details that mine does, and it is in a super cute color! I do not mind that this bag may be made out of a different material, since I did pay such a low price for it. I do not plan on returning it, because my friend loves it! Buyers, beware that you may not be getting the same quality bag as Fjallraven sells, but the quality is still good, and comparable to a Jansport backpack. I would still recommend this bag, as it is super cute, affordable on Amazon, and very trendy nowadays."

class StrTest(unittest.TestCase):
    """Tests for extract_keyword_str"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, extract_keyword_str, None)
        self.assertRaises(TypeError, extract_keyword_str, 0)
        self.assertRaises(TypeError, extract_keyword_str, [])
    
    def test_empty_string(self):
        self.assertEqual(len(extract_keyword_str("")), 0)
        self.assertEqual(extract_keyword_str(""), [])

    def test_sample(self):
        # Correct types
        self.assertGreater(len(extract_keyword_str(sample)), 0)
        self.assertIsInstance(extract_keyword_str(sample), list)
        self.assertIsInstance(extract_keyword_str(sample)[0], tuple)
        self.assertIsInstance(extract_keyword_str(sample)[0][0], str)
        self.assertIsInstance(extract_keyword_str(sample)[0][1], float)
        # The following assume the usage of the NLTK stoplist, which should be default
        self.assertIn(("definitely less waterproof", 8.5), extract_keyword_str(sample))
        self.assertEqual(extract_keyword_str(sample)[0], ("definitely less waterproof", 8.5)) # correct sorting
        # Using the SmartStoplist, the above keyword should be missing
        self.assertGreater(len(extract_keyword_str(sample, useNLTK=False)), 0)
        self.assertNotIn(("definitely less waterproof", 8.5), extract_keyword_str(sample, useNLTK=False))
        self.assertGreater(len(extract_keyword_str(sample)), len(extract_keyword_str(sample, useNLTK=False)))

class ListTest(unittest.TestCase):
    """Tests for extract_keyword_list"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, extract_keyword_list, None)
        self.assertRaises(TypeError, extract_keyword_list, 0)
        self.assertRaises(TypeError, extract_keyword_list, "")

    def test_empty_list(self):
        self.assertEqual(len(extract_keyword_list([])), 0)
        self.assertEqual(extract_keyword_list([]), [])

    def test_sample(self):
        extract_keyword_list([sample])
        self.assertEqual(extract_keyword_str(sample), extract_keyword_list([sample]))
        self.assertEqual(extract_keyword_list([sample]), extract_keyword_list([sample, ""]))
        self.assertNotEqual(extract_keyword_list([sample]), extract_keyword_list([sample, sample]))
        self.assertGreater(len(extract_keyword_list([sample, sample])), len(extract_keyword_list([sample])))
        self.assertEqual(set(extract_keyword_list([sample])), set(extract_keyword_list([sample, sample])))

if __name__ == "__main__":
    unittest.main()
