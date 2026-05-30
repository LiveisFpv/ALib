package handlers

import (
	"VKR_gateway_service/internal/app"
	"VKR_gateway_service/internal/domain"
	"VKR_gateway_service/internal/transport/http/presenters"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GetAuthorProfile
// @Summary Get author ORCID profile
// @Description Returns the authenticated user's optional ORCID author link.
// @Tags author-profile
// @Produce json
// @Success 200 {object} presenters.AuthorProfileResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /author-profile/orcid [get]
func GetAuthorProfile(ctx *gin.Context, a *app.App) {
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	profile, err := a.AuthorProfileService.Get(ctx.Request.Context(), userID)
	if err != nil {
		ctx.JSON(mapGRPCToHTTP(err), presenters.Error(authorProfileUserError(err)))
		return
	}
	ctx.JSON(http.StatusOK, mapAuthorProfile(profile))
}

// UpdateAuthorProfile
// @Summary Save author ORCID profile
// @Description Saves and manually confirms the authenticated user's ORCID author link.
// @Tags author-profile
// @Accept json
// @Produce json
// @Param data body presenters.AuthorProfileUpdateRequest true "ORCID payload"
// @Success 200 {object} presenters.AuthorProfileResponse
// @Failure 400 {object} presenters.ErrorResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 409 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /author-profile/orcid [put]
func UpdateAuthorProfile(ctx *gin.Context, a *app.App) {
	var in presenters.AuthorProfileUpdateRequest
	if err := ctx.ShouldBindJSON(&in); err != nil {
		ctx.JSON(http.StatusBadRequest, presenters.Error(fmt.Errorf("Enter ORCID and confirm authorship.")))
		return
	}
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	profile, err := a.AuthorProfileService.Update(ctx.Request.Context(), userID, &domain.AuthorProfileUpdateInput{
		Orcid:             in.Orcid,
		ConfirmAuthorship: in.ConfirmAuthorship,
	})
	if err != nil {
		ctx.JSON(mapGRPCToHTTP(err), presenters.Error(authorProfileUserError(err)))
		return
	}
	ctx.JSON(http.StatusOK, mapAuthorProfile(profile))
}

// DeleteAuthorProfile
// @Summary Delete author ORCID profile
// @Description Removes the authenticated user's ORCID author link.
// @Tags author-profile
// @Produce json
// @Success 204
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /author-profile/orcid [delete]
func DeleteAuthorProfile(ctx *gin.Context, a *app.App) {
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	if err := a.AuthorProfileService.Delete(ctx.Request.Context(), userID); err != nil {
		ctx.JSON(mapGRPCToHTTP(err), presenters.Error(authorProfileUserError(err)))
		return
	}
	ctx.Status(http.StatusNoContent)
}

// ListAuthorPapers
// @Summary List catalog papers by linked ORCID
// @Description Lists catalog papers whose author ORCID is linked to the authenticated user.
// @Tags author-profile
// @Produce json
// @Success 200 {object} presenters.PapersResponse
// @Failure 401 {object} presenters.ErrorResponse
// @Failure 502 {object} presenters.ErrorResponse
// @Router /author-profile/papers [get]
func ListAuthorPapers(ctx *gin.Context, a *app.App) {
	userID, statusCode, err := resolveUserID(ctx, 0)
	if err != nil {
		ctx.JSON(statusCode, presenters.Error(err))
		return
	}
	papers, err := a.AuthorProfileService.ListPapers(ctx.Request.Context(), userID)
	if err != nil {
		ctx.JSON(mapGRPCToHTTP(err), presenters.Error(authorProfileUserError(err)))
		return
	}
	items := make([]presenters.Paper, 0, len(papers))
	for _, paper := range papers {
		items = append(items, mapPaper(paper))
	}
	ctx.JSON(http.StatusOK, presenters.PapersResponse{Papers: items})
}

func authorProfileUserError(err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return fmt.Errorf("Could not process ORCID profile. Try again later.")
	}

	message := strings.ToLower(st.Message())
	switch st.Code() {
	case codes.InvalidArgument:
		if strings.Contains(message, "checksum") {
			return fmt.Errorf("ORCID checksum is invalid. Check the last character.")
		}
		if strings.Contains(message, "format") {
			return fmt.Errorf("ORCID must use the format 0000-0000-0000-000X.")
		}
		if strings.Contains(message, "confirmation") {
			return fmt.Errorf("Confirm authorship before saving ORCID.")
		}
		if strings.Contains(message, "required") {
			return fmt.Errorf("Enter ORCID.")
		}
		return fmt.Errorf("Check the ORCID value and try again.")
	case codes.AlreadyExists:
		return fmt.Errorf("This ORCID is already linked to another account.")
	case codes.Unauthenticated:
		return fmt.Errorf("Sign in again to link your ORCID.")
	case codes.PermissionDenied:
		return fmt.Errorf("You do not have permission to update this ORCID profile.")
	case codes.Unavailable, codes.DeadlineExceeded, codes.Internal:
		return fmt.Errorf("Author profile service is unavailable. Try again later.")
	default:
		return fmt.Errorf("Could not process ORCID profile. Try again later.")
	}
}

func mapAuthorProfile(profile *domain.AuthorProfile) presenters.AuthorProfileResponse {
	if profile == nil {
		return presenters.AuthorProfileResponse{}
	}
	return presenters.AuthorProfileResponse{
		Orcid:       profile.Orcid,
		Confirmed:   profile.Confirmed,
		ConfirmedAt: profile.ConfirmedAt,
		PaperCount:  profile.PaperCount,
	}
}
