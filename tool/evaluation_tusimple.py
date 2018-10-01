import json
import yaml
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import pylab
from lane import LaneEval
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

overall_acc, overall_fp, overall_fn = 0., 0., 0.
json_pred = [yaml.safe_load(line) for line in open('pred_0555.json').readlines()]
json_gt = [yaml.safe_load(line) for line in open('sort_data_0531.json')]
pred, gt = json_pred[0], json_gt[0]
pred_SUM, gt_SUM = pred['SUM'], gt['SUM']
# Start the list of 'lanes'
for i, j in zip(pred_SUM, gt_SUM):
	pred_lanes = i['lanes']
	gt_lanes = j['lanes']
	y_samples = j['h_samples']
	raw_file = j['raw_file']
	current_folder = re.split('/', raw_file)[2]
	current_filename = 'TuSimple @' + re.split('/', raw_file)[1] + '/' + re.split('/', raw_file)[2]
	img = plt.imread(raw_file)

	if(len(pred_lanes) == 0):
		gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
		img_vis = img.copy()
		for lane in gt_lanes_vis:
			for pt in lane:
				cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))

		print (0.0, 1.0, 1.0)
		overall_acc += 0.0
		overall_fp += 1.0
		overall_fn += 1.0

		cv2.putText(img_vis, 'Accuracy = 0.0 %', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, 'FP = 1.0', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, 'FN = 1.0', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, '%s' % (current_filename), (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		
		plt.imshow(img_vis)
		# plt.imsave(current_folder, img_vis)
		# plt.show()
		plt.close()

		
		#raise Exception('We do not get the predictions of all the test tasks')

	else: 
		gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
		pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
		img_vis = img.copy()

		for lane in gt_lanes_vis:
			cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)
		for lane in pred_lanes_vis:
			cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(255,0,0), thickness=2)

		print LaneEval.bench(pred_lanes, gt_lanes, y_samples)
		a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples)

		cv2.putText(img_vis, 'Accuracy = %.2f' % (round(a, 4) * 100) + ' %', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, 'FP = %.2f' % p, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, 'FN = %.2f'% n, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_vis, '%s' % (current_filename), (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		plt.imshow(img_vis)
		# plt.imsave(current_folder, img_vis)
		# plt.show()
		plt.close()
	
		np.random.shuffle(pred_lanes)
		# Overall Accuracy, False Positive Rate, False Negative Rate
		overall_acc += a
		overall_fp += p
		overall_fn += n
num = len(gt_SUM)
num_pred = len(pred_SUM)
print num
print num_pred
print 'Overall Accuracy', (overall_acc / num)
print 'Overall FP', (overall_fp / num)
print 'Overall FN', (overall_fn / num)
