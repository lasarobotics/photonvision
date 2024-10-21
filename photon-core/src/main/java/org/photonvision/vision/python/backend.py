import torch
import sys
import pathlib
sys.path.append('./model')
pathlib.WindowsPath = pathlib.PosixPath

model = torch.hub.load('./model/', 'custom', path='./model/model.pt', source='local')


def detect(image):
	result = model(image)
	bboxes = []
	for pred in result.xyxy[0].cpu().numpy():
		bboxes.append({
			'x': pred[0],
			'y': pred[1],
			'w': pred[2] - pred[0],
			'h': pred[3] - pred[1],
			'class': pred[5],
			'confidence': pred[4],
		})

	return bboxes
