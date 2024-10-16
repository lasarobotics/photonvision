import grpc
import photonvision_remote_pipeline_pb2 as prp
import photonvision_remote_pipeline_pb2_grpc as prp_grpc
import cv2

current_id = 0

def send_image(image_path):
	im = cv2.imread(image_path)
	channel = grpc.insecure_channel('localhost:50051')
	stub = prp_grpc.ObjectDetectionServiceStub(channel)

	image = cv2.imread(image_path)
	request = prp.ObjectDetectionRequest(image=prp.Image(
		id=5, image_data=bytes(cv2.imencode('.bmp', image)[1]), format=prp.ImageFormat.FORMAT_BMP))
	response = stub.ObjectDetection(request)
	print(response)
