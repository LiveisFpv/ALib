package service

import (
	pb "VKR_gateway_service/gen/go"
	"VKR_gateway_service/internal/domain"
	"VKR_gateway_service/internal/transport/rpc"
	"context"
	"strconv"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc/metadata"
)

type AuthorProfileService interface {
	Get(ctx context.Context, userID int64) (*domain.AuthorProfile, error)
	Update(ctx context.Context, userID int64, input *domain.AuthorProfileUpdateInput) (*domain.AuthorProfile, error)
	Delete(ctx context.Context, userID int64) error
	ListPapers(ctx context.Context, userID int64) ([]*domain.Paper, error)
}

type authorProfileService struct {
	SemanticClient rpc.SemanticClient
	logger         *logrus.Logger
}

func NewAuthorProfileService(semanticClient rpc.SemanticClient, logger *logrus.Logger) AuthorProfileService {
	return &authorProfileService{
		SemanticClient: semanticClient,
		logger:         logger,
	}
}

func (s *authorProfileService) Get(ctx context.Context, userID int64) (*domain.AuthorProfile, error) {
	resp, err := s.SemanticClient.GetMyAuthorProfile(withUserMetadata(ctx, userID), &pb.AuthorProfileRequest{})
	if err != nil {
		s.logger.WithError(err).WithField("user_id", userID).Error("semantic GetMyAuthorProfile RPC failed")
		return nil, err
	}
	return mapAuthorProfile(resp), nil
}

func (s *authorProfileService) Update(ctx context.Context, userID int64, input *domain.AuthorProfileUpdateInput) (*domain.AuthorProfile, error) {
	req := &pb.AuthorProfileUpdateRequest{
		Orcid:             input.Orcid,
		ConfirmAuthorship: input.ConfirmAuthorship,
	}
	resp, err := s.SemanticClient.UpsertMyAuthorProfile(withUserMetadata(ctx, userID), req)
	if err != nil {
		s.logger.WithError(err).WithField("user_id", userID).Error("semantic UpsertMyAuthorProfile RPC failed")
		return nil, err
	}
	return mapAuthorProfile(resp), nil
}

func (s *authorProfileService) Delete(ctx context.Context, userID int64) error {
	_, err := s.SemanticClient.DeleteMyAuthorProfile(withUserMetadata(ctx, userID), &pb.AuthorProfileRequest{})
	if err != nil {
		s.logger.WithError(err).WithField("user_id", userID).Error("semantic DeleteMyAuthorProfile RPC failed")
	}
	return err
}

func (s *authorProfileService) ListPapers(ctx context.Context, userID int64) ([]*domain.Paper, error) {
	resp, err := s.SemanticClient.ListMyAuthorPapers(withUserMetadata(ctx, userID), &pb.AuthorProfileRequest{})
	if err != nil {
		s.logger.WithError(err).WithField("user_id", userID).Error("semantic ListMyAuthorPapers RPC failed")
		return nil, err
	}
	papers := make([]*domain.Paper, 0, len(resp.GetPapers()))
	for _, paper := range resp.GetPapers() {
		papers = append(papers, mapPaperResponse(paper))
	}
	return papers, nil
}

func withUserMetadata(ctx context.Context, userID int64) context.Context {
	return metadata.AppendToOutgoingContext(ctx, "x-user-id", strconv.FormatInt(userID, 10))
}

func mapAuthorProfile(resp *pb.AuthorProfileResponse) *domain.AuthorProfile {
	if resp == nil {
		return &domain.AuthorProfile{}
	}
	return &domain.AuthorProfile{
		Orcid:       resp.GetOrcid(),
		Confirmed:   resp.GetConfirmed(),
		ConfirmedAt: resp.GetConfirmedAt(),
		PaperCount:  resp.GetPaperCount(),
	}
}

func mapPaperResponse(resp *pb.PaperResponse) *domain.Paper {
	if resp == nil {
		return &domain.Paper{}
	}
	return &domain.Paper{
		Id:               resp.GetID(),
		Title:            resp.GetTitle(),
		Abstract:         resp.GetAbstract(),
		Year:             int(resp.GetYear()),
		Best_oa_location: resp.GetBestOaLocation(),
		State:            resp.GetState(),
		ReferencedWorks:  cloneStringSlice(resp.GetReferencedWorks()),
		RelatedWorks:     cloneStringSlice(resp.GetRelatedWorks()),
		CitedByCount:     int(resp.GetCitedByCount()),
		Authors:          cloneStringSlice(resp.GetAuthors()),
		Institutions:     cloneStringSlice(resp.GetInstitutions()),
		Identifiers:      mapPaperIdentifiers(resp.GetIdentifiers()),
	}
}
