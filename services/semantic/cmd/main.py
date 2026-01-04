import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.al_models.e5.encoder import EncoderConfig, SemanticEncoder
from src.config.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_LORA_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_DOC_IDS_PATH,
    FAISS_INDEX_PATH,
    LOG_LEVEL,
    LOGSTASH_HOST,
    LOGSTASH_PORT,
    SEMANTIC_PORT,
)
from src.http.grpc.grpc_server import SemanticServiceGrpc
from src.lib.logger import Logger
from src.services.search.faiss_index import FaissIndex
from src.services.search.faiss_searcher import FaissSearcher
from src.storage.paper_repository import PaperRepository
from src.storage.author_repository import AuthoryRepository
from src.storage.chat_repository import ChatRepository
from src.storage.user_repository import UserRepository
from src.storage.institution_repository import InstitutionRepository
from src.services.search.search_service import SearchService
from src.services.user_service import UserService
from src.services.chat_service import ChatService


def main() -> None:
    logger = Logger(LOGSTASH_HOST, LOGSTASH_PORT, "Semantic_Search_Service", LOG_LEVEL)

    encoder_cfg = EncoderConfig(
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        lora_path=EMBEDDING_LORA_PATH,
    )
    encoder = SemanticEncoder(encoder_cfg)
    index = FaissIndex(index_path=FAISS_INDEX_PATH, doc_ids_path=FAISS_DOC_IDS_PATH)
    paper_repository = PaperRepository()
    searcher = FaissSearcher(encoder, index, paper_repository)
    search_service = SearchService(searcher)

    # ! TODO services, connections and repoes
    user_repo=UserRepository()
    chat_repo=ChatRepository()
    author_repo=AuthoryRepository()
    institution_repo=InstitutionRepository()

    user_service=UserService(user_repo)
    chat_service=ChatService(chat_repo,user_service)


    service = SemanticServiceGrpc(search_service,chat_service,user_service, logger)
    service.serve(SEMANTIC_PORT)


if __name__ == "__main__":
    main()
