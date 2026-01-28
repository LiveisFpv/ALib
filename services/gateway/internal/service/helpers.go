package service

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/domain"
)

// func requestContext(ctx *gin.Context, a *app.App) (context.Context, context.CancelFunc) {
// 	rctx := ctx.Request.Context()
// 	if a != nil && a.Config != nil && a.Config.GRPCTimeout > 0 {
// 		return context.WithTimeout(rctx, a.Config.GRPCTimeout)
// 	}
// 	return rctx, func() {}
// }

// func mapGRPCToHTTP(c codes.Code) int {
// 	switch c {
// 	case codes.InvalidArgument:
// 		return http.StatusBadRequest
// 	case codes.NotFound:
// 		return http.StatusNotFound
// 	case codes.DeadlineExceeded:
// 		return http.StatusGatewayTimeout
// 	case codes.Unavailable:
// 		return http.StatusBadGateway
// 	case codes.PermissionDenied, codes.Unauthenticated:
// 		return http.StatusForbidden
// 	default:
// 		return http.StatusBadGateway
// 	}
// }

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
