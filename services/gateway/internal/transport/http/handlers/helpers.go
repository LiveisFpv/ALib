package handlers

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/transport/http/presenters"
	"context"
	"fmt"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func mapGRPCToHTTP(c codes.Code) int {
	switch c {
	case codes.InvalidArgument:
		return http.StatusBadRequest
	case codes.NotFound:
		return http.StatusNotFound
	case codes.DeadlineExceeded:
		return http.StatusGatewayTimeout
	case codes.Unavailable:
		return http.StatusBadGateway
	case codes.PermissionDenied, codes.Unauthenticated:
		return http.StatusUnauthorized
	default:
		return http.StatusBadGateway
	}
}

func requestContext(ctx *gin.Context, a *app.App) (context.Context, context.CancelFunc) {
	rctx := ctx.Request.Context()
	if a != nil && a.Config != nil && a.Config.GRPCTimeout > 0 {
		return context.WithTimeout(rctx, a.Config.GRPCTimeout)
	}
	return rctx, func() {}
}

func parsePathInt64(ctx *gin.Context, name string) (int64, error) {
	raw := ctx.Param(name)
	if raw == "" {
		return 0, fmt.Errorf("%s path param is required", name)
	}
	return parsePositiveInt64(raw, name)
}

func parseOptionalQueryInt64(ctx *gin.Context, name string) (int64, error) {
	raw := ctx.Query(name)
	if raw == "" {
		return 0, nil
	}
	return parsePositiveInt64(raw, name)
}

func parsePositiveInt64(raw, field string) (int64, error) {
	val, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || val <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", field)
	}
	return val, nil
}

func resolveUserID(ctx *gin.Context, userID int64) (int64, int, error) {
	authID, ok := authUserID(ctx)
	if userID > 0 {
		if ok && authID != userID {
			return 0, http.StatusForbidden, fmt.Errorf("user_id does not match token")
		}
		return userID, 0, nil
	}
	if ok {
		return authID, 0, nil
	}
	return 0, http.StatusUnauthorized, fmt.Errorf("user_id is required")
}

func authUserID(ctx *gin.Context) (int64, bool) {
	val, ok := ctx.Get("user_id")
	if !ok {
		return 0, false
	}
	switch v := val.(type) {
	case int64:
		return v, true
	case int:
		return int64(v), true
	case float64:
		return int64(v), true
	case string:
		id, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0, false
		}
		return id, true
	default:
		return 0, false
	}
}

func authorizeChatAccess(ctx *gin.Context, a *app.App, userID, chatID int64) bool {
	req := &pb.UserChatsReq{UserId: userID}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.GetUserChats(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithField("user_id", userID).Error("AI GetUserChats RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return false
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return false
	}
	for _, chat := range resp.GetChats() {
		if chat.GetChatId() == chatID {
			return true
		}
	}
	ctx.JSON(http.StatusForbidden, presenters.Error(fmt.Errorf("chat access denied")))
	return false
}

func mapChat(chat *pb.Chat) presenters.ChatResponse {
	if chat == nil {
		return presenters.ChatResponse{}
	}
	return presenters.ChatResponse{
		ChatId:    chat.GetChatId(),
		UserId:    chat.GetUserId(),
		UpdatedAt: chat.GetUpdatedAt(),
		Title:     chat.GetTitle(),
	}
}

func mapPapers(papers []*pb.PaperResponse) []presenters.Paper {
	out := make([]presenters.Paper, 0, len(papers))
	for _, p := range papers {
		out = append(out, presenters.Paper{
			Id:               p.GetID(),
			Title:            p.GetTitle(),
			Abstract:         p.GetAbstract(),
			Year:             int(p.GetYear()),
			Best_oa_location: p.GetBestOaLocation(),
		})
	}
	return out
}
