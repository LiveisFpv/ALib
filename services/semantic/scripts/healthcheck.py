import os
import sys
import grpc


def main():
    port = str(os.getenv("SEMANTIC_PORT", "5104"))
    # inside container checking localhost is fine
    host = os.getenv("HEALTH_HOST", "127.0.0.1")
    target = f"{host}:{port}"
    try:
        channel = grpc.insecure_channel(target)
        grpc.channel_ready_future(channel).result(timeout=float(os.getenv("HEALTH_TIMEOUT", "5")))
        sys.exit(0)
    except Exception as e:
        print(f"healthcheck failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

