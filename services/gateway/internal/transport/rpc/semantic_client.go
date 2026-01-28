package rpc

import (
	"context"
	"time"

	pb "VKR_gateway_service/gen/go"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Kept to allow inversion and easier replacement/mocking if needed.
type SemanticClient interface {
	pb.SemanticServiceClient
}

// NewSemanticClient dials the AI gRPC service and returns the client and underlying connection.
// Caller is responsible for closing the returned connection.
func NewSemanticService(ctx context.Context, addr string, timeout time.Duration, opts ...grpc.DialOption) (SemanticClient, *grpc.ClientConn, error) {
	dctx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		dctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	base := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	if len(opts) > 0 {
		base = append(base, opts...)
	}

	conn, err := grpc.DialContext(dctx, addr, base...)
	if err != nil {
		return nil, nil, err
	}
	return pb.NewSemanticServiceClient(conn), conn, nil
}
