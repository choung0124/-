import csv
import re
import codecs
import sys

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Get the CSV file name from the command line argument
csv_file = sys.argv[1]

# Set up your YouTube API key
api_key = 'AIzaSyAhe0mLJS2PYbJQAJ2jI1edRIiuOubZUNw'
youtube = build('youtube', 'v3', developerKey=api_key)

# Initialize an empty list to store comments
exporting_comments = []

# Read the CSV file
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        channel_id = row[1]
        channel_name = row[0]
        print(channel_id, channel_name)

        # Initialize a variable to count comments for the current channel
        channel_comment_count = 0

        # Retrieve the playlist items (videos) for the current channel
        playlist_items_response = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=f'UU{channel_id[2:]}',
            maxResults=50
        ).execute()

        # Extract video IDs from the playlist items
        video_ids = [item['contentDetails']['videoId'] for item in playlist_items_response['items']]

        # Loop through the video IDs
        for video_id in video_ids:
            # Get video details and statistics
            video_response = youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()

            # Get the comment count for the current video
            comment_count = int(video_response['items'][0]['statistics']['commentCount'])
            if comment_count == 0:
                continue

            # Retrieve comments until there are no more comments left
            next_page_token = None
            while True:
                comments_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=1000,
                    pageToken=next_page_token
                ).execute()

                # Loop through the comments
                for comment in comments_response['items']:
                    # Extract the comment text
                    new_comment = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    # Find Korean characters in the comment text
                    korean_comment = re.findall('[ㄱ-ㅎㅏ-ㅣ가-힣]+', new_comment)
                    # Add the Korean comment to the list
                    exporting_comments.append(korean_comment)
                    # Increment the channel comment count
                    channel_comment_count += 1

                    # Check for replies in the comment
                    if 'replies' in comment:
                        # Loop through the replies
                        for reply in comment['replies']['comments']:
                            # Extract the reply text
                            new_reply = reply['snippet']['textDisplay']
                            # Find Korean characters in the reply text
                            korean_reply = re.findall('[ㄱ-ㅎㅏ-ㅣ가-힣]+', new_reply)
                            # Add the Korean reply to the list
                            exporting_comments.append(korean_reply)
                            # Increment the channel comment count
                            channel_comment_count +=1

                # Check if there is a next page of comments
                if 'nextPageToken' in comments_response:
                    next_page_token = comments_response['nextPageToken']
                else:
                    break

        # Print the number of comments crawled for the current channel
        print(f"Number of comments crawled for channel {channel_name}: {channel_comment_count}")

# Write the extracted Korean comments to a text file
with codecs.open("extracted_text_korean_only.txt", "w", "euc-kr") as file:
    for comment in exporting_comments:
        file.write(' '.join(comment) + '\n')

# Print the total number of comments extracted
print(len(exporting_comments))
