"""Camera abstraction layer for Linux using mplayer and video decoding.

The Capture class provided from this module uses mplayer to decode a movie.

Author: Bjoern Barz
"""

import cv2
from PIL import Image
import VideoTools
from scipy.misc import imread
import glob

class Capture(object):
    """Provides access to video devices."""

    def __init__(self, mode, requested_cam_size=(640,480), url=None, videodir=None, videofile=None):

        if mode=='download':
            if url:
                videofile, videolength = VideoTools.download_video( url )
                mode = 'decode'
            else:
                raise Exception("url unspecified")
        else:
            if videofile:
                videolength = VideoTools.get_video_length(videofile)
                mode = 'decode'
            else:
                raise Exception("videofile unspecified")

        if mode=='decode':
            self.frames = VideoTools.decode_video(videofile, videolength)
        else:
            if videodir:
                self.frames = glob.glob("%s/*.png" % (videodir))
            else:
                raise Exception("videodir unspecified")

        self.currentFrame = 0

    def __del__(self):
        pass

    def grabFrame(self):
        """Returns a snapshot from the device as PIL.Image.Image object."""
        
        data, w, h, orientation = self.grabRawFrame()

        return Image.fromstring("RGB", (w, h), data, "raw", "BGR", 0, orientation)


    def grabRawFrame(self):
        """Returns a snapshot from this device as raw pixel data.
        
        This function returns a 4-tuple consisting of the raw pixel data as string,
        the width and height of the snapshot and it's orientation, which is either
        1 (top-to-bottom) or -1 (bottom-to-top).
        """
        
        result, cimg = self.capture.read() # cimg will represent the image as numpy array
        cimg = imread( self.frames[self.currentFrame] )
        height, width, depth = cimg.shape
        
        if self.currentFrame < len(self.frames)-1:
            self.currentFrame = self.currentFrame + 1

        return cimg.tostring(), width, height, 1


    @staticmethod
    def enumerateDevices():
        devices = ()
        return devices
