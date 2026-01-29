package handlers

import (
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/domain"
	"VKR_gateway_service/internal/transport/http/presenters"
	"net/http"

	"github.com/gin-gonic/gin"
)

// PaperAdd
// @Summary Add paper
// @Description Add a paper to the index
// @Tags paper
// @Accept json
// @Produce json
// @Param data body presenters.AddPaperRequest true "Paper data"
// @Success 200 {object} presenters.Paper
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 403 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /ai/paper/add [post]
func PaperAdd(ctx *gin.Context, a *app.App) {
	var in presenters.AddPaperRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}
	paper := &domain.Paper{
		Id:               in.Id,
		Title:            in.Title,
		Abstract:         in.Abstract,
		Year:             in.Year,
		Best_oa_location: in.Best_oa_location,
	}
	reference := make([]*domain.ReferencedPaper, 0, len(in.ReferencedPapers))
	for _, p := range in.ReferencedPapers {
		reference = append(reference, &domain.ReferencedPaper{
			Id: p.Id,
		})
	}
	relate := make([]*domain.RelatedPaper, 0, len(in.RelatedPaper))
	for _, p := range in.RelatedPaper {
		relate = append(relate, &domain.RelatedPaper{
			Id: p.Id,
		})
	}
	paper, err := a.PaperService.AddPaper(ctx, paper, reference, relate)
	if err != nil {
		ctx.JSON(mapGRPCToHTTP(err), presenters.Error(err))
		return
	}

	ctx.JSON(http.StatusOK, presenters.Paper{
		Id:               paper.Id,
		Title:            paper.Title,
		Abstract:         paper.Abstract,
		Year:             paper.Year,
		Best_oa_location: paper.Best_oa_location,
		State:            paper.State,
	})
}
