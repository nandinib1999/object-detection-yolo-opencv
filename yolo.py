import cv2
import numpy as np 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img, starting_time): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

	elapsed_time = time.time() - starting_time
	if args.webcam:
		fps = frame_id / elapsed_time
		print(fps)
	cv2.imshow("Image", img)

def image_detect(img_path): 
	starting_time = time.time()
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image, starting_time)

def webcam_detect():
	global frame_id
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	starting_time = time.time()
	frame_id = 0
	while True:
		_, frame = cap.read()
		frame_id += 1
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame, starting_time)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


if __name__ == '__main__':
	webcam = args.webcam
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	else:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
		cv2.waitKey(0)
	cv2.destroyAllWindows()