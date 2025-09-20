from src.lib.logger import Logger
from src.http.grpc import service_pb2, service_pb2_grpc
from src.services.search.search import SearchService
import grpc
import math


class SemanticServiceHandlerGrpc(service_pb2_grpc.SemanticServiceServicer):
    def __init__(self, search_service: SearchService, logger: Logger):
        self.logger = logger
        self.search_service = search_service

    def SearchPaper(self, request: service_pb2.SearchRequest, context: grpc.ServicerContext) -> service_pb2.PapersResponse:
        self.logger.info(f"SearchPaper request: {request.Input_data}")
        try:
            matching_papers = self.search_service.search_paper(request.Input_data)

            def to_str(v):
                try:
                    return "" if v is None else str(v)
                except Exception:
                    return ""

            def to_int(v):
                try:
                    if v is None:
                        return 0
                    if isinstance(v, float) and math.isnan(v):
                        return 0
                    return int(v)
                except Exception:
                    try:
                        return int(float(v))
                    except Exception:
                        return 0

            papers_resp = service_pb2.PapersResponse()
            for paper in matching_papers:
                papers_resp.Papers.append(service_pb2.PaperResponse(
                    ID=to_str(paper.ID),
                    Title=to_str(paper.Title),
                    Abstract=to_str(paper.Abstract),
                    Year=to_int(paper.Year),
                    Best_oa_location=to_str(paper.Best_oa_location)
                ))
            return papers_resp
        except Exception as e:
            self.logger.error("SearchPaper failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return service_pb2.PapersResponse()

    def AddPaper(self, request: service_pb2.AddRequest, context: grpc.ServicerContext) -> service_pb2.ErrorResponse:
        self.logger.info(f"AddPaper request: {request.Title}")

        # Placeholder for future implementation
        paper = service_pb2.PaperResponse(
            ID=request.ID,
            Title=request.Title,
            Abstract=request.Abstract,
            Year=request.Year,
            Best_oa_location=request.Best_oa_location
        )

        return service_pb2.ErrorResponse(
            Error=""
        )

