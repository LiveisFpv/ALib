import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.config import LOG_LEVEL, LOGSTASH_HOST, LOGSTASH_PORT, SEMANTIC_PORT
from src.lib.logger import Logger
from src.al_models.bert.al_entity import Bert
from src.http.grpc.grpc_server import SemanticServiceGrpc
from src.services.search.search import SearchService

# Init logger for project
logger=Logger('',0,"Semantic_Search_Service",LOG_LEVEL)

# Start AL services
vector_directory = "./data/vectorized/openAlex/"
bert = Bert("intfloat/multilingual-e5-large", vector_directory,logger)
search_service = SearchService(bert,[])

# Start server with logger and Al services
service = SemanticServiceGrpc(search_service,logger)
service.serve(SEMANTIC_PORT)
