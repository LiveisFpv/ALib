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
        def safe_str(v):
            try:
                return "" if pd.isna(v) else str(v)
            except Exception:
                return ""

        def safe_int(v):
            try:
                if pd.isna(v):
                    return 0
            except Exception:
                pass
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return 0

        data = {}
        if not os.path.isdir(folder_path):
            return data
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                for _, row in df.iterrows():
                    paper_id = safe_str(row.get('ID'))
                    if not paper_id:
                        continue
                    data[paper_id] = {
                        "Title": safe_str(row.get("Title", "")),
                        "Abstract": safe_str(row.get("Abstract", "")),
                        "Year": safe_int(row.get("Year", 0)),
                        "Best_oa_location": safe_str(row.get("Best_oa_location", ""))
                    }
        return data

    def search_paper(self,text:str)->list[PaperModel]:
        def safe_str(v):
            try:
                return "" if pd.isna(v) else str(v)
            except Exception:
                return ""

        def safe_int(v):
            try:
                if pd.isna(v):
                    return 0
            except Exception:
                pass
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return 0

        texts = self.semantic_model.get_similar_texts(text)
        relevant = []
        for t in texts:
            paper_id = safe_str(t[0])
            info = self.papers_data.get(paper_id, {})
            paper = PaperModel(
                paper_id,
                Title=safe_str(info.get("Title", "Unknown")),
                Abstract=safe_str(info.get("Abstract", "No abstract")),
                Year=safe_int(info.get("Year", 0)),
                Best_oa_location=safe_str(info.get("Best_oa_location", "No link"))
            )
            relevant.append(paper)
        return relevant
