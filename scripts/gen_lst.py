import glob
import os
import numpy as np

'''
This Script is used for generate flow-first-images.txt, flow-second-images.txt, flow-outputs.txt

Input: video foler, with file name like XXXX.png
Output: flow-first-images.txt, flow-second-images.txt, flow-outputs.txt
'''

video_folder = '/disk1/xueyan-data/test_video/'
video_name = 'tennis'
load_pth = video_folder + video_name + '/'
out_folder = '/home/xueyan/flownet2/results/' + video_name + '/'
data_folder = '/home/xueyan/flownet2/datasets/'

if not os.path.exists(out_folder):
	os.makedirs(out_folder)

file_names = glob.glob(load_pth + "*.jpg", recursive=True)
file_names = sorted(file_names)

pth_lst = []
for i in range(0, len(file_names)-1):
	idx = file_names[i].split('/')[5][:-4]
	data = [file_names[i], file_names[i+1], out_folder + idx + '.flo']
	pth_lst.append(data)

np.save(data_folder + video_name + '.npy', pth_lst)