import os
import sys
from argparse import ArgumentParser
from selenium import webdriver
import time


parser = ArgumentParser(description="Scrape YouTube for data.")
parser.add_argument("--search-file", "-f",
                    type=str,
                    help="Input file containing search times")
parser.add_argument("--output-dir", "-o",
                    type=str,
                    help="The output directory to store collected files.",
                    default="data")


def load_search_file(path):
    with open(path) as f:
        terms = f.readlines()

    return terms


def get_search_results(browser: webdriver.Chrome, query: str):
    url = "https://www.youtube.com/results?search_query={}&sp=CAMSBAgDEAE%253D".format(query)
    browser.get(url)

    for _ in range(4):
        browser.execute_script("window.scrollBy(0, 1080);")
        time.sleep(1)

    elts = browser.find_elements_by_css_selector("ytd-video-renderer")
    for elt in elts:
        print(elt)
        link = elt.find_element_by_id("thumbnail").get_property("href")
        video_id = link.split("?v=")[1]
        break


def scrape(search_terms, output_dir):
    browser = webdriver.Chrome(executable_path="./chromedriver.exe")

    for term in search_terms:
        seaerch_results = get_search_results(browser, term)
        break


if __name__ == "__main__":
    args = vars(parser.parse_args())

    if not args["search_file"]:
        print("Missing required argument: search-file")

    if not os.path.exists(args["search_file"]):
        print("File not found:", args["search_file"])
        sys.exit(1)

    os.makedirs(args["output_dir"], exist_ok=True)

    search_terms = load_search_file(args["search_file"])

    scrape(search_terms, args["output_dir"])
