from src.utils.reviewextractor import check_product_urls, get_amazon_product_id, remove_html_code, get_amazon_link, extract_reviews_for_products
import unittest

sample_web_pages = ['https://www.amazon.com/-/de/dp/B07RF1XD36/ref=lp_16225009011_1_6',
                'https://www.amazon.com/dp/B08JQKMFFB/ref=sspa_dk_detail_2?psc=1&pd_rd_i=B08JQKMFFB&pd_rd_w=5AdCg' +
                '&pf_rd_p=45e679f6-d55f-4626-99ea-f1ec7720af94&pd_rd_wg=bWbE5&pf_rd_r=HJV72D1QHGE2XJ8QJBV0&pd_rd_r' +
                '=b3a4c265-2d13-454f-a385-3ad0a71737eb&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzN1NSWjVRTFFINUFNJmVuY3J5cHR' +
                'lZElkPUEwMjY3OTk2MUQ5ODYwVU4zNlhBVCZlbmNyeXB0ZWRBZElkPUEwMzMwMjc2M1VQMVJXVllMVVpGJndpZGdldE5hbWU9c' +
                '3BfZGV0YWlsJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==',
                'https://www.amazon.com/product-reviews/B08KH53NKR/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar' +
                '=all_stars&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar'
                ]
false_web_pages = ['https://www.amazon.com/-/de',
                   'https://www.amazon.com/-/de/dp/B07RF1XD362/ref=lp_16225009011_1_6']

htmltext = [
    'Test <br><br> 123\n\n456<br>'

]


class AmazonUrlTest(unittest.TestCase):
    def test_url_check(self):
        self.assertRaises(TypeError, check_product_urls, None)
        self.assertFalse(check_product_urls([]))
        self.assertTrue(check_product_urls(sample_web_pages))
        self.assertRaises(Exception, check_product_urls, false_web_pages)

    def test_product_id_extraction(self):
        self.assertRaises(AttributeError, get_amazon_product_id, None)
        self.assertTrue(get_amazon_product_id(sample_web_pages[0]) == 'B07RF1XD36')
        self.assertTrue(get_amazon_product_id(sample_web_pages[1]) == 'B08JQKMFFB')
        self.assertTrue(get_amazon_product_id(sample_web_pages[2]) == 'B08KH53NKR')
        self.assertRaises(Exception, check_product_urls, false_web_pages[0])

    def test_html_cleaner(self):
        self.assertRaises(AttributeError, remove_html_code, None)
        self.assertTrue(remove_html_code(htmltext[0]) == 'Test 123 456')

    def test_review_url_creation(self):
        self.assertRaises(TypeError, get_amazon_link, None)

    def test_review_extraction(self):
        self.assertTrue(len(extract_reviews_for_products(sample_web_pages, 1, 3)) == 3)
        self.assertTrue(len(extract_reviews_for_products([], 1, 3)) == 0)
        self.assertRaises(Exception,extract_reviews_for_products, false_web_pages, 1, 3)


if __name__ == "__main__":
    unittest.main()

