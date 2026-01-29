package service

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/domain"
)

func mapChat(chat *pb.Chat) *domain.Chat {
	if chat == nil {
		return &domain.Chat{}
	}
	return &domain.Chat{
		ChatId:    chat.GetChatId(),
		UserId:    chat.GetUserId(),
		UpdatedAt: chat.GetUpdatedAt(),
		Title:     chat.GetTitle(),
	}
}

func mapPapers(papers []*pb.PaperResponse) []*domain.Paper {
	out := make([]*domain.Paper, 0, len(papers))
	for _, p := range papers {
		out = append(out, &domain.Paper{
			Id:               p.ID,
			Title:            p.Title,
			Abstract:         p.Abstract,
			Year:             int(p.Year),
			Best_oa_location: p.BestOaLocation,
			State:            p.State,
		})
	}
	return out
}
