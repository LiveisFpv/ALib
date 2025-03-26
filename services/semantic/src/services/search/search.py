from domain.models.paper import PaperModel
from domain.models.search import SearchResultModel

class Search:
    def __init__(self,semantic_model,graph_model):
        self.semantic_model=semantic_model
        self.graph_model=graph_model

    
