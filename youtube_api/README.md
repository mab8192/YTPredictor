

# API Scraper
While this is not *scraping* in the proper sense of the word, we are pulling API
data until we get the necessary data for the project. Also, instead of wrapping the
`requests` library I found a client repo officially licensed by YT, linked below. This
was used for sanitation/validation purposes.

## Setup
```bash
cd <path_to_project>
# only run this if you don't have a venv setup
virtualenv venv  
# make sure the requirements are installed,  `google-api-python-client` is listed there
pip install -r requirements.txt
mkdir -p ./youtube_api/thumbnails
touch ./youtube_api/data.json
```

## API Key
You can get an API key by going [here](https://developers.google.com/youtube/v3/getting-started)
and following instructions 1 - 3. I did not do the OAuth portion. Once you have an API key,
just hard set the `AUTH_KEY` value in `scrape_youtube.py` or whatever you want. Then
change to the `youtube_api/` directory and run
```bash
echo "AUTH_KEY = <INSERT_API_KEY_HERE>" > ./config.py
```

## Running
Simply run
```bash
# assuming your venv is still activated
python3 ./youtube_api/scrape_youtube.py
```
and it will save json data to `youtube_api/data.json` and thumbnails in `youtube_api/thumbnails/` in whatever
format was specified in the url.

## Additional Links
- [Youtube API V3](https://developers.google.com/youtube/v3/docs)
- [Youtube API Client](https://github.com/googleapis/google-api-python-client)