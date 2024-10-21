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
		results = backend.detect(cv2.imdecode(np.fromstring(request.image.image_data, np.uint8), 0))
		response = prp.ObjectDetectionResponse(results = prp.DetectionResults(
			id=5,
			results=[prp.BoundingBox(x=int(r['x']), y=int(r['y']), width=int(r['w']), height=int(r['h']), cls=int(r['class']), confidence=float(r['confidence']))
					for r in results]
		))
		print(int(results[0]['class']))
		print(results[0]['class'])
		with open('recv.png', 'wb') as f:
			f.write(request.image.image_data)
		return response

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	prp_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionService(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	print("Server is running on port 50051")
	server.wait_for_termination()

if __name__ == '__main__':
	serve()
