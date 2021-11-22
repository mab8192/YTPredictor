from googleapiclient.discovery import build
from datetime import timedelta as td
from datetime import datetime as dt
from config import AUTH_KEY
import requests
import time
import json


service = build('youtube', 'v3', developerKey=AUTH_KEY)
THUMBNAIL_SIZE = 'default'  # possible options are 'default' for 120x90, 'medium' for 320x180, 'high' for 480x360


def get_dt_from_today(days):
    # return date in dumb RFC 3339 standard thing
    return (dt.today() - td(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')


START_DATE = get_dt_from_today(days=7)
END_DATE = get_dt_from_today(days=6)


def search_videos(next_page=None):
    search = service.search()
    search_params = {'part': 'snippet', 'publishedAfter': START_DATE, 'publishedBefore': END_DATE, 'maxResults': 50}
    if next_page:
        search_params['pageToken'] = next_page
    query = search.list(**search_params)
    return query.execute()


def get_video_stats(video_set):
    videos = service.videos()
    ids = ','.join(x['id']['videoId'] for x in video_set)
    search_params = {'part': 'statistics', 'id': ids}
    query = videos.list(**search_params)
    return query.execute()


def collect_data(search_results, statistics):
    ids = set(x['id']['videoId'] for x in search_results['items']).union(set(x['id'] for x in statistics['items']))
    data = {x: {} for x in ids}
    search_result_keys = ('publishedAt', 'channelId', 'title', 'description', 'channelTitle', 'publishTime')
    for result in search_results['items']:
        data[result['id']['videoId']] = {x: result['snippet'][x] for x in search_result_keys}
        data[result['id']['videoId']]['thumbnail'] = result['snippet']['thumbnails'][THUMBNAIL_SIZE]
    for result in statistics['items']:
        data[result['id']].update(result['statistics'])
    return data


def pull_thumbnails(page_data):
    for _id, dataset in page_data.items():
        video_resp = requests.get(dataset['thumbnail']['url'])
        filename = f'{_id}.{dataset["thumbnail"]["url"].split(".")[-1]}'
        with open(f'./thumbnails/{filename}', 'wb') as f:
            f.write(video_resp.content)


def dump_json(page_data):
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        data = {}
    data.update(page_data)
    with open('data.json', 'w') as f:
        json.dump(data, f)


def get_data():
    next_page = None
    try:
        while True:
            search_results = search_videos(next_page)
            statistics = get_video_stats(search_results['items'])
            next_page = search_results['nextPageToken']
            page_data = collect_data(search_results, statistics)
            dump_json(page_data)
            pull_thumbnails(page_data)
            time.sleep(1)
    except Exception as e:
        print(f'The following error occurred while gathering data: {e}. Terminating')
    service.close()


if __name__ == '__main__':
    get_data()
