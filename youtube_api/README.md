

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
. venv/bin/activate
# make sure the requirements are installed,  `google-api-python-client` is listed there
pip install -r requirements.txt
mkdir -p ./youtube_api/thumbnails
touch ./youtube_api/data.json
```

## API Key
You can get an API key by going [here](https://developers.google.com/youtube/v3/getting-started)
and following instructions 1 - 3. I did not do the OAuth portion. Once you have an API key,
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

## Data
As mentioned, the data is saved in the json format with the following key-value pairs (top-level keys are the youtube video ids)
```json
}
'zBcKq_Zj-TQ': {   'channelId': 'UCNIiH_4ArJNd_cDZApZ7AFg',
                       'channelTitle': 'TVING',
                       'commentCount': '483',
                       'description': '이선빈X한선화X정은지 그리고 최시원의 본격_기승전술_드라마 '
                                      '[술꾼도시여자들]   오직 티빙에서 스트리밍하세요! 티빙 바로가기 '
                                      '...',
                       'dislikeCount': '159',
                       'favoriteCount': '0',
                       'likeCount': '8560',
                       'publishTime': '2021-11-15T09:00:34Z',
                       'publishedAt': '2021-11-15T09:00:34Z',
                       'thumbnail': {   'height': 90,
                                        'url': 'https://i.ytimg.com/vi/zBcKq_Zj-TQ/default.jpg',
                                        'width': 120},
                       'title': '[술꾼도시여자들] 🔥흑화한 한선화, 불법 개 농장 운영 박영규에 도끼로 살벌 '
                                '복수⛏',
                       'viewCount': '1058593'},
 ...
}
```
To load the data in a dictionary format into var `data`,
```python
import json
with open('./youtube_api/data.json', 'r') as f:
    data = json.load(f)
```
Thumbnails are stored in `youtube_api/thumbnails/<video_id>.<some_format>`.


## Additional Links
- [Youtube API V3](https://developers.google.com/youtube/v3/docs)
- [Youtube API Client](https://github.com/googleapis/google-api-python-client)
