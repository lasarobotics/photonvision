import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('port')

	args = parser.parse_args()

import grpc
from concurrent import futures
import photonvision_remote_pipeline_pb2 as prp
import photonvision_remote_pipeline_pb2_grpc as prp_grpc
import traceback
import cv2
import backend
import numpy as np

class ObjectDetectionService(prp_grpc.ObjectDetectionServiceServicer):
	def ObjectDetection(self, request, context):
		results = backend.detect(cv2.imdecode(np.frombuffer(request.image.image_data, np.uint8), 0))
		response = prp.ObjectDetectionResponse(results = prp.DetectionResults(
			id=5,
			results=[prp.BoundingBox(x=int(r['x']), y=int(r['y']), width=int(r['w']), height=int(r['h']), cls=int(r['class']), confidence=float(r['confidence']))
					for r in results]
		))
		with open('recv.png', 'wb') as f:
			f.write(request.image.image_data)
		return response

def serve(port: int):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	prp_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionService(), server)
	server.add_insecure_port(f'[::]:{port}')
	server.start()
	print(f"Server is running on port {port}")
	server.wait_for_termination()

if __name__ == '__main__':
	serve(int(args.port))
