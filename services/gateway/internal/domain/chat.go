package domain

type Chat struct {
	ChatId    int64  `json:"chat_id"`
	UserId    int64  `json:"user_id"`
	UpdatedAt string `json:"updated_at"`
	Title     string `json:"title"`
}

type ChatHistoryMessage struct {
	SearchQuery string   `json:"search_query"`
	CreatedAt   string   `json:"created_at"`
	Papers      []*Paper `json:"papers"`
}
