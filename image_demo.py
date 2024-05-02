
import cv2
import numpy as np
import os
import torch


def putText(canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
	# Draw the border
	cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
				color=borderColor, thickness=thickness+2, lineType=lineType)
	# Draw the text
	cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
				color=textColor, thickness=thickness, lineType=lineType)
 
 
dataset = 'path/to/your/dataset'
anchor_img = cv2.imread(os.path.join(dataset, '0.jpg'))
test_img = cv2.imread(os.path.join(dataset, '1.jpg'))

anchor_img = cv2.resize(anchor_img, (1280, 1280))
test_img = cv2.resize(test_img, (1280, 1280))

width, height = anchor_img.shape[1], anchor_img.shape[0]


from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat()

# #Simple inference with batch = 1
# output_anchor = xfeat.detectAndCompute(anchor_img, top_k = 4096)[0]
# print("----------------")
# print("keypoints: ", output_anchor['keypoints'].shape)
# print("descriptors: ", output_anchor['descriptors'].shape)
# print("scores: ", output_anchor['scores'].shape)
# print("----------------\n")



# output_test = xfeat.detectAndCompute(test_img, top_k = 4096)[0]
# print("----------------")
# print("keypoints: ", output_test['keypoints'].shape)
# print("descriptors: ", output_test['descriptors'].shape)
# print("scores: ", output_test['scores'].shape)
# print("----------------\n")


mkpts_0, mkpts_1 = xfeat.match_xfeat(anchor_img, test_img)
points1, points2 = mkpts_0, mkpts_1

H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4.0, maxIters=700, confidence=0.995)

inliers = inliers.flatten() > 0
print('Number of inliers:', np.sum(inliers))

if H.dtype != np.float32:
    H = H.astype(np.float32)
    
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
 
warped_image = cv2.warpPerspective(test_gray, H, (width, height))
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
# Create a new image with the width being the sum of both images' widths and the height being the maximum of both images' heights
new_width = anchor_img.shape[1] + warped_image.shape[1]
new_height = max(anchor_img.shape[0], warped_image.shape[0])

# Make a new canvas, filled with zeros (black)
combined_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

# Place the anchor image on the left
combined_image[:anchor_img.shape[0], :anchor_img.shape[1]] = anchor_img

# Place the warped image on the right
combined_image[:warped_image.shape[0], anchor_img.shape[1]:anchor_img.shape[1]+warped_image.shape[1]] = warped_image



kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

# Draw matches
matched_frame = cv2.drawMatches(anchor_img, kp1, test_img, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)

color = (240, 89, 169)

# Add a colored rectangle to separate from the top frame
cv2.rectangle(matched_frame, (2, 2), (width*2-2, height-2), color, 5)

t_font = cv2.FONT_HERSHEY_SIMPLEX
t_font_scale = 0.9
t_line_type = cv2.LINE_AA
t_line_color = (0,255,0)
t_line_thickness = 3

# Adding captions on the top frame canvas
putText(canvas=matched_frame, text="%s Matches: %d"%('xfeat', len(good_matches)), org=(10, 30), fontFace=t_font, 
	fontScale=t_font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=t_line_type)

cv2.imshow('Matched Frame', matched_frame)
# Display the result
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)



