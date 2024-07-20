import os, sys
import pickle, zstd
import numpy as np

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder

class Applet(AtlasApplet):
    MODEL_WIDTH=640
    MODEL_HEIGHT=640

    def __init__(self):
        # self.orig_shape = orig_shape
        super().__init__(port=5600)
    
    def pre_process(self,blob):
        image = ImageEncoder.bytes_to_image(blob)
        image = image.astype(np.float32)
        image = image / 255
        '''Transposing must go here, after jpeg decoding by cv2'''
        image = image.transpose(2, 0, 1).tobytes()

        return image

    def post_process(self,result_list):
        print(type(result_list[0]))
        blob = pickle.dumps(result_list)       
        blob = zstd.compress(blob, 1)

        return [blob]

if __name__ == '__main__':
    '''Set the original shape of images from you applications here.'''
    #orig_shape = (720, 1280, 3)
    Applet().run()
