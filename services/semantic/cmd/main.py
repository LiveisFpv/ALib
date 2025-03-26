import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.config import LOG_LEVEL, LOGSTASH_HOST, LOGSTASH_PORT
from src.lib.logger import Logger
from src.al_models.bert.al_entity import Model
from src.http.grpc.grpc_server import SemanticServiceGrpc

# Init logger for project
logger=Logger(LOGSTASH_HOST,LOGSTASH_PORT,"Semantic_Search_Service",LOG_LEVEL)

# Start AL services
vector_directory = "./data/vectorized/openAlex/"
bert = Model("intfloat/multilingual-e5-large", vector_directory,logger)

# Start server with logger and Al services
service = SemanticServiceGrpc(bert,logger)
service.serve()