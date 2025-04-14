from src.al_models.bert.al_entity import Bert
from src.domain.models.paper import PaperModel
import pandas as pd
import os

class SearchService:
    def __init__(self,semantic_model:Bert,graph_model):
        self.semantic_model=semantic_model
        self.graph_model=graph_model
        self.papers_data = self.load_papers_data("data/processed/openAlex")
    
    def load_papers_data(self, folder_path):
        data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                for _, row in df.iterrows():
                    paper_id = str(row['ID'])
                    data[paper_id] = {
                        "Title": row.get("Title", ""),
                        "Abstract": row.get("Abstract", ""),
                        "Year": row.get("Year", 0),
                        "Best_oa_location": row.get("Best_oa_location", "")
                    }
        return data

    def search_paper(self,text:str)->list[PaperModel]:
        texts = self.semantic_model.get_similar_texts(text)
        relevant = []
        for text in texts:
            paper_id = str(text[0])
            info = self.papers_data.get(paper_id, {})
            paper = PaperModel(
                paper_id,
                Title=info.get("Title", "Unknown"),
                Abstract=info.get("Abstract", "No abstract"),
                Year=info.get("Year", 0),
                Best_oa_location=info.get("Best_oa_location", "No link")
            )
            relevant.append(paper)
        return relevant