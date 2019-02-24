from visual_helper import read, computeColor
import numpy as np
import cv2
import math
import glob
import os

truerange = 1
range_f = truerange * 1.04

flo_foler = '/home/xueyan/flownet2/results/'

dir_lst = [x[0] for x in os.walk(flo_foler)][1:]

for i in range(0, len(dir_lst)):
	flo_foler = dir_lst[i]
	data_name = dir_lst[i].split('/')[5]
	print(data_name)
	out_foler = '/home/xueyan/flownet2/visual/' + data_name + '/'

	if not os.path.exists(out_foler):
	    os.makedirs(out_foler)

	file_names = glob.glob(flo_foler + '/*.flo', recursive=True)
	file_names = sorted(file_names)

	for i in range(0, len(file_names)):
		flow = read(file_names[i])

		out_name = out_foler + file_names[i].split('/')[6][:-4] + '.png'
		u = flow[: , : , 0]
		v = flow[: , : , 1]		

		img = computeColor(u/truerange, v/truerange)
		h,w,c = img.shape

		img[int(round(h/2)),:,:] = 0
		img[:,int(round(w/2)),:] = 0

		# color encoding scheme for optical flow
		img = computeColor(u/range_f/math.sqrt(2), v/range_f/math.sqrt(2));
		cv2.imwrite(out_name, img)