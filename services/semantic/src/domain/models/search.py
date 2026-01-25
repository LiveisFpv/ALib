class SearchResultModel:
    def __init__(self, id, file_path, source_file, similarity):
        self.id = id
        self.file_path = file_path
        self.source_file = source_file
        self.similarity = similarity