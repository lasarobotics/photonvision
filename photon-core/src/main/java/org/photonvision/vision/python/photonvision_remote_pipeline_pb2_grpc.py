# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import photonvision_remote_pipeline_pb2 as photonvision__remote__pipeline__pb2

GRPC_GENERATED_VERSION = '1.66.2'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in photonvision_remote_pipeline_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ObjectDetectionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ObjectDetection = channel.unary_unary(
                '/ObjectDetectionService/ObjectDetection',
                request_serializer=photonvision__remote__pipeline__pb2.ObjectDetectionRequest.SerializeToString,
                response_deserializer=photonvision__remote__pipeline__pb2.ObjectDetectionResponse.FromString,
                _registered_method=True)


class ObjectDetectionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ObjectDetection(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ObjectDetection': grpc.unary_unary_rpc_method_handler(
                    servicer.ObjectDetection,
                    request_deserializer=photonvision__remote__pipeline__pb2.ObjectDetectionRequest.FromString,
                    response_serializer=photonvision__remote__pipeline__pb2.ObjectDetectionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ObjectDetectionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('ObjectDetectionService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ObjectDetectionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ObjectDetection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ObjectDetectionService/ObjectDetection',
            photonvision__remote__pipeline__pb2.ObjectDetectionRequest.SerializeToString,
            photonvision__remote__pipeline__pb2.ObjectDetectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)