import math
import re
import threading
import time
from typing import List

import requests
from selenium import webdriver

from chromedriver_py import binary_path

# requires Google Chrome Version 88

review_part_start = '<span class="cr-original-review-content">'
review_part_end = "</span>"

title_part_start = (
    '<span id="productTitle" class="a-size-large product-title-word-break">'
)
title_part_end = "</span>"


class AmazonReviewPageExtractor:
    def __init__(self, page_content: str):
        """
        extracts all reviews for a html page content
        :param page_content: source of the web page
        """
        self.data = page_content  # raw page data
        self.reviews = list()  # list of reviews
        self.extracts_reviews()  # extract reviews

    def extracts_reviews(self) -> None:
        """
        extract the reviews from raw data to the reviews list
        Each review consists of a heading and the body.
        The text of the review is in a span of the class="cr-original-review-content"
        The reviews can be extracted with this property
        :return: None
        """
        review_parts = self.data.count(review_part_start)  # count review tokens
        if review_parts > 0:
            start_idx = self.data.find(review_part_start)  # starting point
            end_idx = self.data.find(review_part_end, start_idx)  # starting end point
            while start_idx != -1:  # As long as there are still reviews
                # extract the header an find the body
                header = (
                    remove_html_code(
                        self.data[start_idx + len(review_part_start) : end_idx]
                    )
                    + ". "
                )
                start_idx = self.data.find(review_part_start, end_idx)
                end_idx = self.data.find(review_part_end, start_idx)
                # extract the body
                content = remove_html_code(
                    self.data[start_idx + len(review_part_start) : end_idx]
                )
                start_idx = self.data.find(review_part_start, end_idx)
                end_idx = self.data.find(review_part_end, start_idx)
                # concat the header and the body, store into the review array
                self.reviews.append(header + content)


def remove_html_code(text: str) -> str:
    """
    used to remove al html special character
    :param text: review text
    :return: cleaned up review text
    """
    result = re.sub(" +", " ", text.replace("<br>", " ").replace("\n", " ").strip())
    return result


def get_amazon_link(product_id: str, page: int) -> str:
    """
    Generates an amazon review link for the following parameters
    :param product_id: unique id of the product
    :param page: review page number
    :return: amazon review page without any filter
    """
    result = (
        "https://www.amazon.com/-/de/product-reviews/"
        + product_id
        + "/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar=all_stars&reviewerType=all_reviews&pageNumber="
        + str(page)
    )
    return result


def get_amazon_product_id(url: str) -> str:
    """
    Extract the amazon product id from product or review web page
    :param url: url of product
    :return: product id
    """
    start = url.find("/dp/")  # search pattern for a product url
    count = 4
    if start == -1:
        start = url.find("/product-reviews/")  # search pattern for a review page
        count = 17
        if start == -1:
            start = url.find("/product/")  # search pattern for a review page
            count = 9
            if start == -1:
                raise Exception("Failed to find the product id in the given url: " + url)
    end = url.find("/", start + count)
    if end == -1:
        end = url.find("?", start + count)
        if end == -1:
            end = len(url)
    result = url[start + count : end]
    return result


def get_chrome_options() -> webdriver.ChromeOptions:
    """
    Chrome Option definition
    :return: Default driver options
    """
    options = webdriver.ChromeOptions()
    if not bool(__debug__):
        options.add_argument("--headless")  # background task; don't open a window
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return options


def extract_amazon_review(thread_idx: int, product_id: str, data: list):
    """
    thread function
    creates a driver and opens the url
    stores the reviews into shared memory data
    :param thread_idx: index of current thread
    :param product_id: product id from amazon, used to generate the review url
    :param data: shared memory array to save the web page source
    :return: None
    """
    driver = webdriver.Chrome(
        options=get_chrome_options(), executable_path=binary_path
    )  # create driver engine
    driver.get(
        get_amazon_link(product_id, thread_idx + 1)
    )  # set browser to use this page
    if bool(__debug__):
        time.sleep(60)
    else:
        time.sleep(6)  # wait for page
    result = driver.page_source  # extract page html source
    driver.quit()  # close driver
    review_page = AmazonReviewPageExtractor(result)  # initialize review extractor class
    data[thread_idx] = review_page.reviews  # store extracted reviews in shared memory


def extract_all_amazon_reviews(
    product_url: str, max_pages: int, max_threads: int
) -> list:
    """
    main function to extract all amazon reviews for a given product url
    :param product_url: product url copied from web browser
    :param max_pages: page limit if there are more review pages
    :param max_threads: thread limit used for html extraction
    :return: array of reviews
    """
    product_id = get_amazon_product_id(product_url)
    thread_data: List[str] = [""] * max_pages
    threads: List[threading.Thread] = list()
    last_review_page = max_pages
    for thread_iter in range(0, max_pages, max_threads):
        # create only a limited amount of threads
        for web_page in range(thread_iter, thread_iter + max_threads, 1):
            # create thread and start
            threads.append(
                threading.Thread(
                    target=extract_amazon_review,
                    args=(web_page, product_id, thread_data),
                )
            )
            threads[web_page].start()
        for web_page in range(thread_iter, thread_iter + max_threads, 1):
            # join threads and count the number of reviews
            threads[web_page].join()
            review_list = thread_data[web_page]
            number_of_reviews = len(review_list)
            if number_of_reviews == 0:
                last_review_page = web_page
                break
        if last_review_page != max_pages:
            break
    pages = thread_data[:last_review_page]
    # concat reviews
    result = list()
    for page in pages:
        result += page
    return result


def check_product_urls(urls: List[str]) -> bool:
    """
    Used to check the input amazon urls
    :param urls: array of amazon product urls
    :return: True if everything is fine and raises an exception if there was no product id
    """
    if len(urls) == 0:
        return False
    else:
        for url_idx in range(len(urls)):
            url = urls[url_idx]
            product_id = get_amazon_product_id(url)
            if len(product_id) != 10:
                # After a random test, all product IDs have 10 characters
                raise Exception(
                    "Failed to extract the product id for the "
                    + str(url_idx + 1)
                    + ". url: "
                    + url
                )
        return True


def thread_extract_reviews_for_product(
    product_url: str, max_pages: int, max_threads: int, thread_idx: int, data: list
):
    """
    Thread function used to extract the reviews for a given product url
    The first 3 parameters are passed on to extract_all_amazon_reviews and the thread_idx is used to
    store the result into the shared memory data array
    :param product_url: given product url
    :param max_pages: page limit
    :param max_threads: thread limit for execution
    :param thread_idx: id of the current thread
    :param data: shared memory, used to store the result of each thread from extract_all_amazon_reviews
    :return: None
    """
    data[thread_idx] = extract_all_amazon_reviews(product_url, max_pages, max_threads)


def extract_reviews_for_products(
    products: List[str], max_pages: int, max_threads: int
) -> list:
    """
    Main function
    Is used to extract all reviews from a list of amazon products in parallel.
    The extracted review pages can be limited via max_pages.
    In order to optimize the performance used, the number of threads can be set using max_threads.
    We recommend to use 10 threads per product --> for 3 products pass 30 threads as max_threads
    :param products: list of product urls
    :param max_pages: review page limit for each product
    :param max_threads: thread limit for the execution in parallel
    :return: List[List[Reviews]] for each product an array of reviews
    """
    if check_product_urls(products):
        threads: List[threading.Thread] = list()
        data = [None] * len(products)
        threads_for_product = math.trunc(max_threads / len(products))
        for product_num in range(len(products)):
            url = products[product_num]
            threads.append(
                threading.Thread(
                    target=thread_extract_reviews_for_product,
                    args=(url, max_pages, threads_for_product, product_num, data),
                )
            )
            threads[product_num].start()
        for product_num in range(len(products)):
            threads[product_num].join()
        return data
    else:
        return []


def extract_product_title_and_jpg(url: str) -> (str, bytes):
    """
    Extract the product title and the product image for a given amazon product url
    :param url: url of the product
    :return: title and jpg image as bytes
    """
    driver = webdriver.Chrome(
        options=get_chrome_options(), executable_path=binary_path
    )  # create driver engine
    driver.get(url)  # set browser to use this page
    if bool(__debug__):
        time.sleep(60)
    else:
        time.sleep(6)  # wait for page
    data = driver.page_source  # extract page html source
    start_idx = data.find(title_part_start)  # starting point
    if start_idx != -1:
        end_idx = data.find(title_part_end, start_idx)  # starting end point
        title = remove_html_code(data[start_idx + len(title_part_start) : end_idx])
    else:
        title = "unknown"
    try:
        # Search Pattern for the product image container
        img_url_start = '<div id="main-image-container" class="a-dynamic-image-container"'
        start_offset = data.find(img_url_start)
        # Try to find a src attribute in the container
        start_idx = data.find('src="', start_offset)
        end_idx = data.find('"', start_idx + len('src="'))
        img_url = data[start_idx + len('src="') : end_idx]
        # Send HTTP Request to get the image
        r = requests.get(img_url)  # set browser to use this page
        img = r.content
    except:
        img = ''
    return title, img


if __name__=='__main__':
    url = 'https://www.amazon.com/-/de/dp/B08316YSKH'
    print(get_amazon_product_id(url))