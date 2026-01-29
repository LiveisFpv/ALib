package rpc

import (
	"context"
	"time"

	"google.golang.org/grpc"
)

// UnaryTimeoutInterceptor enforces a default timeout when the caller didn't set a deadline.
func UnaryTimeoutInterceptor(timeout time.Duration) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if timeout <= 0 {
			return invoker(ctx, method, req, reply, cc, opts...)
		}
		if _, ok := ctx.Deadline(); ok {
			return invoker(ctx, method, req, reply, cc, opts...)
		}
		tctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		return invoker(tctx, method, req, reply, cc, opts...)
	}
}

// StreamTimeoutInterceptor enforces a default timeout when the caller didn't set a deadline.
func StreamTimeoutInterceptor(timeout time.Duration) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		if timeout <= 0 {
			return streamer(ctx, desc, cc, method, opts...)
		}
		if _, ok := ctx.Deadline(); ok {
			return streamer(ctx, desc, cc, method, opts...)
		}
		tctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		return streamer(tctx, desc, cc, method, opts...)
	}
}
