import grpc
from concurrent import futures
import photonvision_remote_pipeline_pb2 as prp
import photonvision_remote_pipeline_pb2_grpc as prp_grpc
import traceback
import cv2

class ObjectDetectionService(prp_grpc.ObjectDetectionServiceServicer):
	def ObjectDetection(self, request, context):
		a = prp.ObjectDetectionResponse(results = prp.DetectionResults(
			id=5,
			results=[prp.BoundingBox(x=1, y=1, width=50, height=100, cls=2, confidence=1),
				prp.BoundingBox(x=50, y=50, width=200, height=5, cls=1, confidence=0.58)]
		))
		with open('recv.png', 'wb') as f:
			f.write(request.image.image_data)
		return a

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	prp_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionService(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	print("Server is running on port 50051")
	server.wait_for_termination()

if __name__ == '__main__':
	serve()
