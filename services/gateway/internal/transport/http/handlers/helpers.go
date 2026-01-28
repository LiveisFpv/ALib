package handlers

import (
	"VKR_gateway_service/internal/domain"
	"VKR_gateway_service/internal/transport/http/presenters"
	"fmt"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

func parsePathInt64(ctx *gin.Context, name string) (int64, error) {
	raw := ctx.Param(name)
	if raw == "" {
		return 0, fmt.Errorf("%s path param is required", name)
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

// function that search userID in context and compare it with given ID
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

// function that try to find userID in context and parse it
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

func mapChat(chat *domain.Chat) presenters.ChatResponse {
	if chat == nil {
		return presenters.ChatResponse{}
	}
	return presenters.ChatResponse{
		ChatId:    chat.ChatId,
		UserId:    chat.UserId,
		UpdatedAt: chat.UpdatedAt,
		Title:     chat.Title,
	}
}

func mapPapers(papers []*domain.Paper) []presenters.Paper {
	out := make([]presenters.Paper, 0, len(papers))
	for _, p := range papers {
		out = append(out, presenters.Paper{
			Id:               p.Id,
			Title:            p.Title,
			Abstract:         p.Abstract,
			Year:             p.Year,
			Best_oa_location: p.Best_oa_location,
		})
	}
	return out
}
