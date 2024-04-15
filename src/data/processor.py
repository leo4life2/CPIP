import numpy as np
import cv2
import os
from os.path import join, exists, dirname
import logging
import argparse
import yaml
import sys
import json
from tqdm import tqdm

sys.path.append(dirname(sys.path[0]))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='/mnt/data/CPIP')
    parser.add_argument('-f', '--fov', type=int, default=90)
    parser.add_argument('-y', '--yaw_num', type=int, default=18)
    parser.add_argument('-t', '--height', type=int, default=810)
    parser.add_argument('-w', '--width', type=int, default=1440)
    parser.add_argument('-v', '--video', type=str, default='ICT1_0')
    opt = parser.parse_args()
    return opt

# Equirectangular class definition for handling 360-degree images
class Equirectangular:
    RADIUS = 128

    # Initialize the Equirectangular object with the provided configuration
    def __init__(self, opt):
        self.fov=opt.fov
        self.yaw_num=opt.yaw_num
        self.height=opt.height
        self.width=opt.width

        self.yaw_list = np.linspace(0, 360, opt.yaw_num + 1)[:-1]
        self.FOV = opt.fov
        self.height = opt.height
        self.width = opt.width
        self.wFOV = self.FOV
        self.hFOV = float(self.height) / self.width * self.wFOV
        self.c_x = (self.width - 1) / 2.0
        self.c_y = (self.height - 1) / 2.0
        self.wangle = (180 - self.wFOV) / 2.0
        self.hangle = (180 - self.hFOV) / 2.0
        
        self._compute_geometry()
        self.y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        self.z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    # Compute the geometry of the equirectangular projection
    def _compute_geometry(self):
        self.w_len = 2 * self.RADIUS * np.sin(np.radians(self.wFOV / 2.0)) / np.sin(np.radians(self.wangle))
        self.h_len = 2 * self.RADIUS * np.sin(np.radians(self.hFOV / 2.0)) / np.sin(np.radians(self.hangle))
        self.w_interval = self.w_len / (self.width - 1)
        self.h_interval = self.h_len / (self.height - 1)

    # Create XYZ maps for the equirectangular projection
    def _create_xyz_maps(self):
        x_map = np.full((self.height, self.width), self.RADIUS, np.float32)
        y_map = np.tile((np.arange(0, self.width) - self.c_x) * self.w_interval, [self.height, 1])
        z_map = -np.tile((np.arange(0, self.height) - self.c_y) * self.h_interval, [self.width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.stack([(self.RADIUS / D * x_map), (self.RADIUS / D * y_map), (self.RADIUS / D * z_map)], axis=-1)
        return xyz
    
    # Apply rotation to the XYZ maps using input angles
    def _apply_rotation(self, xyz, THETA, PHI):
        R1, _ = cv2.Rodrigues(self.z_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, self.y_axis) * np.radians(-PHI))
        xyz = xyz.reshape([self.height * self.width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        return xyz

    # Get a perspective image from the equirectangular image using input angles
    def GetPerspective(self, image, THETA):
        equ_h, equ_w, _ = image.shape
        equ_cx, equ_cy = (equ_w - 1) / 2.0, (equ_h - 1) / 2.0
        xyz = self._create_xyz_maps()
        xyz = self._apply_rotation(xyz, THETA, 0)

        lat = np.arcsin(xyz[:, 2] / self.RADIUS)
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        lon, lat = np.degrees(lon).reshape(self.height, self.width), -np.degrees(lat).reshape(self.height, self.width)
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(image, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

# Slice equirectangular video into perspective videos
def slice_data(input_path,gt_path,output_path,scale,opt):
    equ2pers = Equirectangular(opt)
    yaw_list=equ2pers.yaw_list
    with open(gt_path,'r') as f:
        gt=json.load(f)['keyframes']

    index=1

    if not exists(output_path):
        os.makedirs(output_path)
        cap = cv2.VideoCapture(input_path)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
        progress_bar = tqdm(total=total_frames, desc="Processing Video Frames")

        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            index_str=str(index).zfill(5)

            if index_str in gt:
                translation=gt[index_str]['trans']
                rot_base=gt[index_str]['rot']
                if ret == True:
                    for YAW in yaw_list:
                        pers_frame=equ2pers.GetPerspective(frame,YAW)
                        ang = (YAW+rot_base* 180 / np.pi)%360
                        cv2.imwrite(filename=join(output_path,f'{str(index).zfill(5)}_{translation[0]*scale}_{translation[1]*scale}_{ang}.png'),img=pers_frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                else: 
                    break
            index+=1
            progress_bar.update(1)  # Update the progress bar

        progress_bar.close()  # Close the progress bar when done

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    else:
        print('Already exists the folder, please delete it first')
    

# Prepare the dataset by slicing and compressing videos
def main():
    opt=options()
    root = opt.root

    with open(join(root,'measurement_scale.yaml'), 'r') as f:
        scale = yaml.safe_load(f)[opt.video.split('_')[0]]

    ##### Slice database and query equirectangular video to perspective video
    video_path=join(root,'video',opt.video+'.mp4')
    gt_path=join(root,'gt',opt.video+'.json')
    output_path=join(root,'output',opt.video)

    slice_data(video_path,gt_path,output_path,scale,opt)

if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    main()
