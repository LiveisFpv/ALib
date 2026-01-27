package handlers

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/transport/http/presenters"
	"context"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/status"
)

// PaperAdd
// @Summary Add paper
// @Description Add a paper to the index
// @Tags ai
// @Accept json
// @Produce json
// @Param data body presenters.AddPaperRequest true "Paper data"
// @Success 200 {object} map[string]string
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 500 {object} presenters.ErrorResponse
// @Router /ai/paper/add [post]
func PaperAdd(ctx *gin.Context, a *app.App) {
	var in presenters.AddPaperRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(err))
		return
	}

	req := &pb.AddRequest{
		ID:             in.Id,
		Title:          in.Title,
		Abstract:       in.Abstract,
		Year:           int64(in.Year),
		BestOaLocation: in.Best_oa_location,
	}
	if len(in.ReferencedPapers) > 0 {
		req.ReferencedWorks = make([]*pb.ReferencedWorks, 0, len(in.ReferencedPapers))
		for _, r := range in.ReferencedPapers {
			req.ReferencedWorks = append(req.ReferencedWorks, &pb.ReferencedWorks{ID: r.Id})
		}
	}
	if len(in.RelatedPaper) > 0 {
		req.RelatedWorks = make([]*pb.RelatedWorks, 0, len(in.RelatedPaper))
		for _, r := range in.RelatedPaper {
			req.RelatedWorks = append(req.RelatedWorks, &pb.RelatedWorks{ID: r.Id})
		}
	}

	rctx := ctx.Request.Context()
	if a.Config.GRPCTimeout > 0 {
		var cancel context.CancelFunc
		rctx, cancel = context.WithTimeout(rctx, a.Config.GRPCTimeout)
		defer cancel()
	}
	resp, err := a.AI.AddPaper(rctx, req)
	if err != nil {
		if a.Logger != nil {
			a.Logger.WithError(err).WithField("id", in.Id).Error("AI AddPaper RPC failed")
		}
		if s, ok := status.FromError(err); ok {
			ctx.JSON(mapGRPCToHTTP(s.Code()), presenters.Error(fmt.Errorf(s.Message())))
			return
		}
		ctx.JSON(http.StatusBadGateway, presenters.Error(err))
		return
	}
	if msg := resp.GetError(); msg != "" {
		ctx.JSON(http.StatusBadRequest, &presenters.ErrorResponse{Error: msg})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"status": "ok"})
}
