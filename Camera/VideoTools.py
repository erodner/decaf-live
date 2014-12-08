import numpy as np
from scipy.misc import imread
import pafy
import tempfile
import os
import re
import subprocess
import glob

""" downloads a youtube video using pafy and returns the filename and the length of the 
    video in seconds """
def download_video (url, videofile=None, tmproot=None):

    if not videofile: 
        # create a temporary directory
        tempdir = tempfile.mkdtemp(prefix='y2b',dir=tmproot)
        tempvideofile, videofile = tempfile.mkstemp(dir=tempdir, prefix='y2bvideo')
        os.close(tempvideofile)

    video = pafy.new(url)
    print video.title
    print "Duration: ", video.duration

    # now, get the length in seconds, which will be important later on for
    # decoding
    times = video.duration
    times_arr = times.rsplit(':')
    total_seconds = 0
    for t in times_arr:
        total_seconds = 60*total_seconds + int(t)
    print "Duration in seconds: ", total_seconds

    # best = video.getbest()

    # display all possible streams
    print "Possible video streams:"
    for s in video.streams:
        print(s)

    best = video.getbest(preftype="flv")

    print "Downloading the %s video from %s" % (best.resolution, best.url)

    best.download(quiet=False, filepath=videofile)

    return videofile, total_seconds


""" return a list of image filenames, where the images contain the frames of the video """
def decode_video (videofile, videolength, videotmpdir = None):

    # Unfortunately, due to a mysterious bug in mplayer, we have to specify the end position
    mplayer_command = [ 'mplayer', '-vo', 'png:z=9:prefix=y2b', '-nosound', '-sstep', '1', '-endpos', "%d" % (videolength-2), videofile ]

    if videotmpdir is None:
        videotmpdir = os.path.dirname(videofile)
    
    print "Decoding the video with mplayer and storing the images in %s" % (videotmpdir)
    currentdir = os.getcwd()
    try:
        os.chdir( videotmpdir )
        subprocess.call(mplayer_command)
    finally:
        os.chdir( currentdir )

    return sorted(glob.glob("%s/y2b*.png" % (videotmpdir) ))

def get_video_length(videofile):
    # parse for ID_LENGTH 
    mplayeridentifyout = subprocess.check_output( ["mplayer", "-identify", "-frames", "0", videofile] )
    lines = mplayeridentifyout.rsplit('\n')
    for line in lines:
        m = re.match("ID_LENGTH=(\d\.)", line)
        if m:
            timelength = float(m.group(1))
            return timelength

    Exception("Unable to obtain the video length with mplayer")
    return 0
