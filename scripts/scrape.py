import os
import sys
from argparse import ArgumentParser


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


def get_search_results(query):
    # Can't use beautifulsoup, try selenium or scrapy or something
    pass


def scrape(search_terms, output_dir):
    for term in search_terms:
        seaerch_results = get_search_results(term)
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
