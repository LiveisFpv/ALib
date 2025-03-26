class PaperModel:
    def __init__(self, ID, Title, Abstract, Year, Best_oa_location, Referenced_works=[], Related_works=[],Cited_by_count=0):
        self.ID = ID
        self.Title = Title
        self.Abstract = Abstract
        self.Year = Year
        self.Best_oa_location = Best_oa_location
        self.Referenced_works = Referenced_works
        self.Related_works = Related_works
        self.Cited_by_count = Cited_by_count