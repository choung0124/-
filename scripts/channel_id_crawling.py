from googleapiclient.discovery import build
import argparse
import csv

parser = argparse.ArgumentParser(description="Get YouTube channel ID from the username.")
parser.add_argument("--username", "-u", type=str, help="YouTube channel username")
args = parser.parse_args()
ch_username = args.username

def get_channel_id(api_key, query):
    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.search().list(
        part='snippet',
        type='channel',
        q=query,
        maxResults=1
    ).execute()

    if 'items' in response and len(response['items']) > 0:
        return response['items'][0]['snippet']['channelId']
    else:
        return None

if __name__ == "__main__":
    api_key = "your_api_key"
    query = ch_username
    channel_id = get_channel_id(api_key, query)

    if channel_id:
        print(f"Channel ID: {channel_id}")
        with open("channel_ids.csv", "a") as f:

            data = [query, channel_id]

            writer = csv.writer(f)
            writer.writerow(data)

    else:
        print("Channel not found")

