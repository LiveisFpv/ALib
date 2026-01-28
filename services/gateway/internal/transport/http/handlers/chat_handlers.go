package handlers

import (
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/domain"
	"VKR_gateway_service/internal/transport/http/presenters"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
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
// @Failure 502 {object} presenters.ErrorResponse
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
	if len(in.Title) == 0 {
		ctx.JSON(http.StatusBadRequest, presenters.Error(fmt.Errorf("Empty title")))
		return
	}
	chat, err := a.ChatService.CreateChat(ctx, int(userID), in.Title)

	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
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
// @Success 200 {object} presenters.ChatsResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /chats [get]
func GetUserChats(ctx *gin.Context, a *app.App) {
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}

	chats, err := a.ChatService.GetUserChats(ctx, int(userID))

	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}

	out := presenters.ChatsResponse{Chats: make([]presenters.ChatResponse, 0, len(chats))}
	for _, chat := range chats {
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
// @Success 200 {object} presenters.ChatHistoryResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /chats/{chat_id}/history [get]
func GetChatHistory(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	ChatMessages, err := a.ChatService.GetChatHistory(ctx, int(chatID), int(userID))

	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}

	out := presenters.ChatHistoryResponse{ChatMessages: make([]presenters.ChatHistoryMessage, 0, len(ChatMessages))}
	for _, msg := range ChatMessages {
		out.ChatMessages = append(out.ChatMessages, presenters.ChatHistoryMessage{
			SearchQuery: msg.SearchQuery,
			CreatedAt:   msg.CreatedAt,
			Papers:      mapPapers(msg.Papers),
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
// @Param data body presenters.ChatHistoryCreateRequest true "Search query"
// @Success 200 {object} presenters.ChatHistoryMessage
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /chats/{chat_id}/history [post]
func CreateChatHistory(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	var in presenters.ChatHistoryCreateRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	ChatMessage, err := a.ChatService.Search(ctx, in.Text, int(chatID), int(userID))
	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	out := presenters.ChatHistoryMessage{
		SearchQuery: ChatMessage.SearchQuery,
		CreatedAt:   ChatMessage.CreatedAt,
		Papers:      mapPapers(ChatMessage.Papers)}
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
// @Failure 502 {object} presenters.ErrorResponse
// @Router /chats/{chat_id} [put]
func UpdateChat(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	var in presenters.CreateChatRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	chat := &domain.Chat{
		UserId: userID,
		ChatId: chatID,
		Title:  in.Title,
	}
	chat, err = a.ChatService.UpdateChat(ctx, chat)
	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
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
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /chats/{chat_id} [delete]
func DeleteChat(ctx *gin.Context, a *app.App) {
	chatID, err := parsePathInt64(ctx, "chat_id")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	err = a.ChatService.DeleteChat(ctx, int(chatID), int(userID))
	if err != nil {
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	ctx.Status(http.StatusOK)
}
