import grpc
from concurrent import futures
from src.lib.logger import Logger
from src.http.grpc import service_pb2_grpc
from src.services.search.search import SearchService
from src.http.grpc.grpc_handler import SemanticServiceHandlerGrpc

class SemanticServiceGrpc:
    def __init__(self,search_service: SearchService, logger: Logger):
        self.logger = logger
        self.logger.info(f"Starting grpc server")
        self.search_service = search_service
        # Добавляем обработчик
        self.handler = SemanticServiceHandlerGrpc(search_service, logger)

    def serve(self,port=50051):
        """Запуск сервера с определенным портом"""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        service_pb2_grpc.add_SemanticServiceServicer_to_server(self.handler, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.server.start()
        self.logger.info(f"Semantic service started on port: {port}")
        self.server.wait_for_termination()
        self.logger.info(f"Semantic service stoped on port: {port}")