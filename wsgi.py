import os
from diarization_service import app, start_consumer_thread

# Start the RabbitMQ consumer in a background thread
start_consumer_thread()

if __name__ == "__main__":
    # Local development fallback
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)