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
from src.services.search.search import SearchService


def main() -> None:
    logger = Logger(LOGSTASH_HOST, LOGSTASH_PORT, "Semantic_Search_Service", LOG_LEVEL)

    encoder_cfg = EncoderConfig(
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        lora_path=EMBEDDING_LORA_PATH,
    )
    encoder = SemanticEncoder(encoder_cfg)
    index = FaissIndex(index_path=FAISS_INDEX_PATH, doc_ids_path=FAISS_DOC_IDS_PATH)
    repository = PaperRepository()
    searcher = FaissSearcher(encoder, index, repository)
    search_service = SearchService(searcher)

    service = SemanticServiceGrpc(search_service, logger)
    service.serve(SEMANTIC_PORT)


if __name__ == "__main__":
    main()
