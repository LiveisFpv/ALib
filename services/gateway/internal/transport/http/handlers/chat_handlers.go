package handlers

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/transport/http/presenters"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/status"
)

// CreateChat
// @Summary Create chat
// @Description Create a new chat for the user
// @Tags chat
// @Accept json
// @Produce json
// @Param data body presenters.CreateChatRequest true "Chat data"
// @Success 200 {object} presenters.ChatResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats [post]
func CreateChat(ctx *gin.Context, a *app.App) {
	var in presenters.CreateChatRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, in.UserId)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}

	req := &pb.Chat{
		UserId: userID,
		Title:  in.Title,
	}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.CreateNewChat(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithField("user_id", userID).Error("AI CreateNewChat RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	chat := resp.GetChat()
	if chat == nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(fmt.Errorf("empty chat response")))
		return
	}
	ctx.JSON(http.StatusOK, mapChat(chat))
}

// GetUserChats
// @Summary Get user chats
// @Description Get all chats for a user
// @Tags chat
// @Accept json
// @Produce json
// @Param user_id query int false "User ID"
// @Success 200 {object} presenters.ChatsResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats [get]
func GetUserChats(ctx *gin.Context, a *app.App) {
	userID, err := parseOptionalQueryInt64(ctx, "user_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, userID)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}

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
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}

	out := presenters.ChatsResponse{Chats: make([]presenters.ChatResponse, 0, len(resp.GetChats()))}
	for _, chat := range resp.GetChats() {
		out.Chats = append(out.Chats, mapChat(chat))
	}
	ctx.JSON(http.StatusOK, out)
}

// GetChatHistory
// @Summary Get chat history
// @Description Get chat history by chat ID
// @Tags chat
// @Accept json
// @Produce json
// @Param chat_id path int true "Chat ID"
// @Param user_id query int false "User ID"
// @Success 200 {object} presenters.ChatHistoryResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats/{chat_id}/history [get]
func GetChatHistory(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, err := parseOptionalQueryInt64(ctx, "user_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, userID)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	if !authorizeChatAccess(ctx, a, userID, chatID) {
		return
	}

	req := &pb.HistoryReq{ChatId: chatID}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.GetChatHistory(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithField("chat_id", chatID).Error("AI GetChatHistory RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}

	out := presenters.ChatHistoryResponse{ChatMessages: make([]presenters.ChatHistoryMessage, 0, len(resp.GetChatMessages()))}
	for _, msg := range resp.GetChatMessages() {
		out.ChatMessages = append(out.ChatMessages, presenters.ChatHistoryMessage{
			SearchQuery: msg.GetSearchQuery(),
			CreatedAt:   msg.GetCreatedAt(),
			Papers:      mapPapers(msg.GetPapers().GetPapers()),
		})
	}
	ctx.JSON(http.StatusOK, out)
}

// CreateChatHistory
// @Summary Add chat history entry
// @Description Create a new chat history entry by chat ID and search text
// @Tags chat
// @Accept json
// @Produce json
// @Param chat_id path int true "Chat ID"
// @Param user_id query int false "User ID"
// @Param data body presenters.ChatHistoryCreateRequest true "Search query"
// @Success 200 {object} presenters.SearchPaperResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats/{chat_id}/history [post]
func CreateChatHistory(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, err := parseOptionalQueryInt64(ctx, "user_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, userID)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	if !authorizeChatAccess(ctx, a, userID, chatID) {
		return
	}
	var in presenters.ChatHistoryCreateRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}

	req := &pb.SearchRequest{
		InputData: in.Text,
		ChatId:    chatID,
	}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.SearchPaper(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithFields(map[string]interface{}{
				"chat_id": chatID,
				"user_id": userID,
			}).Error("AI Update chat RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}

	out := presenters.SearchPaperResponse{Papers: mapPapers(resp.GetPapers())}
	ctx.JSON(http.StatusOK, out)
}

// UpdateChat
// @Summary Update chat title
// @Description Update chat title by owner (user)
// @Tags chat
// @Accept json
// @Produce json
// @Param chat_id path int true "Chat ID"
// @Param data body presenters.CreateChatRequest true "Chat data"
// @Success 200 {object} presenters.ChatResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats/{chat_id} [put]
func UpdateChat(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, err := parseOptionalQueryInt64(ctx, "user_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, userID)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	if !authorizeChatAccess(ctx, a, userID, chatID) {
		return
	}
	var in presenters.CreateChatRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	req := &pb.UpdateChatReq{
		ChatId: chatID,
		UserId: userID,
		Title:  in.Title,
	}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.UpdateChat(rctx, req)
	if err != nil {
		a.Logger.WithError(err).WithFields(map[string]interface{}{
			"chat_id": chatID,
			"user_id": userID,
		}).Error("Update Chat RPC failed")
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	chat := resp.GetChat()
	if chat == nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(fmt.Errorf("empty chat response")))
		return
	}
	ctx.JSON(http.StatusOK, mapChat(chat))
}

// DeleteChat
// @Summary Delete chat
// @Description Delete chat by owner
// @Tags chat
// @Accept json
// @Produce json
// @Param chat_id path int true "Chat ID"
// @Success 200
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /chats/{chat_id} [delete]
func DeleteChat(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, err := parseOptionalQueryInt64(ctx, "user_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, userID)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	if !authorizeChatAccess(ctx, a, userID, chatID) {
		return
	}
	req := &pb.DeleteChatReq{ChatId: chatID, UserId: userID}
	rctx, cancel := requestContext(ctx, a)
	defer cancel()
	resp, err := a.AI.DeleteChat(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithFields(map[string]interface{}{
				"chat_id": chatID,
				"user_id": userID,
			}).Error("AI Update chat RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	if resp.Error != "" {
		ctx.JSON(http.StatusBadRequest, presenters.Error(fmt.Errorf(resp.Error)))
	}
	ctx.Status(http.StatusOK)
}
