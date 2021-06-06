import subprocess
import urllib
from argparse import ArgumentParser
from os import path
import youtube_dl


def download_video(video_link, target_dir):
    url_data = urllib.parse.urlparse(video_link)
    query = urllib.parse.parse_qs(url_data.query)
    id = query["v"][0]
    file_name = path.join(target_dir, f"{id}.mp4")

    # return_code = subprocess.call(
    #     ["youtube-dl", "https://youtube.com/watch?v={}".format(id), "--quiet", "-f",
    #      "bestvideo[ext={}]+bestaudio/best".format("mp4"), "--output", file_name, "--no-continue"],
    #     stderr=subprocess.DEVNULL)

    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    ydl_opts = {
        'username': r'eitan.kosman',
        'password': 'eitangoogle',
        'format': "bestvideo[ext={}]+bestaudio/best".format("mp4"),
        'no-continue': True,
        'outtmpl': f'{target_dir}/%(id)s',
        'progress_hooks': [my_hook],
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_link])
    except Exception as e:
        pass

    return file_name


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("--target_dir", default="./videos/", help="Where to locate the videos?")
    parser.add_argument("--ann_file", help="Where is the annotation file?")
    parser.add_argument("--idx_to_labels", help="Labels to number txt file")
    args = parser.parse_args()

    with open(args.ann_file, 'r') as fp:
        file_lines = fp.read().splitlines(keepends=False)

    with open(args.idx_to_labels, 'r') as fp:
        idx_2_labels = fp.read().splitlines(keepends=False)

    for line in file_lines:
        line = line.split(' ')
        link = line[0]
        anns = line[1:]
        anns = [idx_2_labels[int(idx)] for idx in anns]
        download_video(link, args.target_dir)
