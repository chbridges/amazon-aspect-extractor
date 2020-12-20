import io
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# test

url = (
    "https://www.amazon.com/-/de/product-reviews/B07GLV1VC7/"
    + "ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar=all_stars"
    + "&reviewerType=all_reviews&pageNumber=1"
)

driver = webdriver.Chrome()


options = Options()
options.add_argument("--headless")  # background task; don't open a window
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")  # I copied this, so IDK?
options.add_argument("--disable-dev-shm-usage")  # this too
driver.get(url)  # set browser to use this page
time.sleep(6)  # let the scripts load
html = driver.page_source  # copy from chrome process to your python instance
filename = "demo.txt"
character_encoding = "utf-8"
with io.open(filename, "w", encoding=character_encoding) as file:
    file.write(html)
driver.quit()
