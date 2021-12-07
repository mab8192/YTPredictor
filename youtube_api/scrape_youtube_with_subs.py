from googleapiclient.discovery import build
from datetime import timedelta as td
from datetime import datetime as dt
import requests
import random
from config import AUTH_KEY
import time
import json


# service instance is a helpful client wrapper of requests specifically built for YouTubes RESTful API v3 endpoints
service = build('youtube', 'v3', developerKey=AUTH_KEY)
THUMBNAIL_SIZE = 'default'  # possible options are 'default' for 120x90, 'medium' for 320x180, 'high' for 480x360


def get_time_slices(hour_ndx):
    """
    We divide time slices into 24 hour widths starting from 7 days ago and return the dates in the RFC 3339 standard
    """
    start_date = (dt.today() - td(days=7) + td(hours=hour_ndx)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = (dt.today() - td(days=7) + td(hours=hour_ndx + 1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return start_date, end_date


def search_videos(next_page=None, start_date=None, end_date=None):
    """
    Search videos using YouTubes RESTful api, calling the 'search' endpoint
    """
    search = service.search()
    search_params = {'part': 'snippet', 'maxResults': 50, 'type': 'video', 'relevanceLanguage': 'en'}
    if next_page:
        search_params['pageToken'] = next_page
    if start_date:
        search_params['publishedAfter'] = start_date
    if end_date:
        search_params['publishedBefore'] = end_date
    query = search.list(**search_params)
    return query.execute()

def get_channels_for_data(channeldata):
    """
    Used to make a quick query and grab the channel subscriber associated with the data.
    """

    channels = service.channels()
    ids = ','.join(x['snippet']['channelId'] for x in channeldata)
    search_params = {'part': 'statistics', 'id': ids}
    query = channels.list(**search_params)
    return query.execute()

        




def get_video_stats(video_set):
    """
    Statistics like views, likes, etc. are not returned by the
    `search` endpoint so we then query the `statistics` endpoint
    """
    videos = service.videos()
    ids = ','.join(x['id']['videoId'] for x in video_set)
    search_params = {'part': 'statistics', 'id': ids}
    query = videos.list(**search_params)
    return query.execute()


def collect_data(search_results, statistics, channel_info):
    """
    Generate a friendly data structure as documented in the README. This is because we return JSON format of RESTful
    API responses that are lists of dicts containing dicts and lists of dicts, etc.
    """
    ids = set(x['id']['videoId'] for x in search_results['items']).union(set(x['id'] for x in statistics['items']))
    data = {x: {} for x in ids}
    search_result_keys = ('publishedAt', 'channelId', 'title', 'description', 'channelTitle', 'publishTime')
    for result in search_results['items']:
        data[result['id']['videoId']] = {x: result['snippet'][x] for x in search_result_keys}
        data[result['id']['videoId']]['thumbnail'] = result['snippet']['thumbnails'][THUMBNAIL_SIZE]
    for result in statistics['items']:
        data[result['id']].update(result['statistics'])
    for item in data:
        for result in channel_info['items']:
            if data[item]['channelId'] == result['id']:
                data[item]['subscriberCount'] = result['statistics']['subscriberCount']
                #if item['channelId'] == result['id']:
                #item['channelStats'] = result
    return data


def pull_thumbnails(page_data):
    """
    We pulled thumbnail URLs above so we now download them and save them to file.
    """
    for _id, dataset in page_data.items():
        video_resp = requests.get(dataset['thumbnail']['url'])
        filename = f'{_id}.{dataset["thumbnail"]["url"].split(".")[-1]}'
        with open(f'./thumbnails/{filename}', 'wb') as f:
            f.write(video_resp.content)


def dump_json(page_data):
    """
    Add data to existing dataset. Because the data is stored along YouTube video key, we are guaranteed unique
    entries so long as the video keys are unique and we add to the same dictionary. Then write to .json file
    for later access.
    """
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        data = {}
    data.update(page_data)
    with open('data.json', 'w') as f:
        json.dump(data, f)


def get_data():
    """
    Main driver of the script to pull data and organize it, clean it, and write it to file
    """
    hours = list(range(24))
    random.shuffle(hours)
    next_page = None
    start_date, end_date = get_time_slices(hours.pop())
    try:
        while True:
            search_results = search_videos(next_page=next_page, start_date=start_date, end_date=end_date)
            statistics = get_video_stats(search_results['items'])
            channel_info = get_channels_for_data(search_results['items'])
            next_page = search_results.get('nextPageToken', None)
            page_data = collect_data(search_results, statistics, channel_info)
            dump_json(page_data)
            pull_thumbnails(page_data)
            if not hours:
                raise ValueError(f'End of day reached. Terminating to maintain 7-day term.')
            if next_page is None:
                start_date, end_date = get_time_slices(hours.pop())
            time.sleep(3)  # sleep to avoid potential api calls / sec quota limits
    except Exception as e:
        print(f'The following error occurred while gathering data: {e}')
    service.close()


if __name__ == '__main__':
    get_data()
