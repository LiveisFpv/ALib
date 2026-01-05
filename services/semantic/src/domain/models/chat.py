class ChatModel:
    def __init__(self,id,user_id,updated_at,title):
        self.id=id
        self.user_id=user_id
        self.updated_at=updated_at
        self.title=title
class ChatMessage:
    def __init__(self,search_query,created_at,papers):
        self.search_query=search_query
        self.created_at=created_at
        self.papers=papers