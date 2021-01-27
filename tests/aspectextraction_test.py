import re
import unittest

from src.utils.aspectextraction import create_aspectmask

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


class CreateAspectMaskTest(unittest.TestCase):
    """Tests for create_aspectmask"""

    def test_invalid_inputs(self):
        self.assertRaises(TypeError, create_aspectmask, None)
        self.assertRaises(TypeError, create_aspectmask, 0)
        self.assertRaises(TypeError, create_aspectmask, [""])

    def test_empty_string(self):
        aspectvector = create_aspectmask("")

        self.assertIsInstance(aspectvector, list)
        self.assertEqual(len(aspectvector), 0)

    def test_sample(self):
        aspectvector = create_aspectmask(sample, return_as_int=False)
        aspectvector_int = create_aspectmask(sample, return_as_int=True)
        aspectvector_adj = create_aspectmask(sample, ignore_adjectives=False)

        sample_tokens = re.split(" |\n", re.sub(r"[^\w\d\s]|'", "", sample))

        self.assertIsInstance(aspectvector, list)
        self.assertGreater(len(aspectvector), 0)
        self.assertEqual(len(aspectvector), len(sample_tokens))
        self.assertIsInstance(aspectvector[0], bool)
        self.assertIn(True, aspectvector)
        self.assertIn(False, aspectvector)

        self.assertEqual(len(aspectvector), len(aspectvector_int))
        self.assertIsInstance(aspectvector_int[0], int)
        self.assertIn(1, aspectvector_int)
        self.assertIn(0, aspectvector_int)
        for i in range(len(aspectvector_int)):
            self.assertEqual(bool(aspectvector_int[i]), aspectvector[i])
            self.assertEqual(aspectvector_int[i], int(aspectvector[i]))

        self.assertEqual(len(aspectvector), len(aspectvector_adj))
        truecount = 0
        truecount_adj = 0
        for i in range(len(aspectvector_adj)):
            if aspectvector[i] is True:
                self.assertEqual(aspectvector_adj[i], True)
                truecount += 1
            if aspectvector_adj[i] == 1:
                truecount_adj += 1
            elif aspectvector_adj[i] == 0:
                self.assertEqual(aspectvector[i], False)
        self.assertLess(truecount, truecount_adj)


if __name__ == "__main__":
    unittest.main()
