import datetime
import os
import json
import uuid
import logging
import tempfile
from pathlib import Path
from threading import Thread
import torch
import boto3
import pika
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Configuration
RABBITMQ_HOST = os.getenv("MQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("MQ_PORT", 5672)
RABBITMQ_USER = os.getenv("MQ_USERNAME", "guest")
RABBITMQ_PASS = os.getenv("MQ_PASSWORD", "guest")
MQ_VOICE_SPEAKER = os.getenv("MQ_VOICE_SPEAKERS", "meeting-voice-speakers")
MQ_VOICE_EXCHANGE = os.getenv("MQ_VOICE_EXCHANGE", "meeting-voice")
MQ_VOICE_RESULT_ROUTING_KEY = os.getenv("MQ_VOICE_RESULT_ROUTING_KEY", "voice-result")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

MODEL_PATH = os.getenv("MODEL_PATH", "./models/speaker-diarization-3.1/config.yaml")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize AWS S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# Device selection for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# Load speaker diarization pipeline locally
pipeline = Pipeline.from_pretrained("./models/pyannote_diarization_config.yaml")
pipeline.to(torch.device("cuda"))


# Initialize Flask app
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify(status="ok"), 200

@app.route("/speaker-diarization", methods=["POST"])
def diarize_file():
    """
    Accepts a file upload via multipart/form-data under 'file' key,
    processes diarization immediately, and returns JSON result.
    """
    if 'file' not in request.files:
        return jsonify(error="Missing file parameter"), 400
    upload = request.files['file']
    if upload.filename == '':
        return jsonify(error="Empty filename"), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        input_path = base / upload.filename
        wav_path = base / "converted.wav"
        # Save uploaded file
        upload.save(str(input_path))
        # Convert and diarize
        convert_to_wav(input_path, wav_path)
        diarization = pipeline(str(wav_path))

        # Collect segments
        data = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            data.append({
                "startTime": format_time(turn.start),
                "endTime": format_time(turn.end),
                "speaker": speaker
            })

    return jsonify({"type": "SPEAKER", "data": data}), 200

def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Convert audio file to mono 16kHz WAV using pydub.
    """
    # Load audio (auto-detect format)
    audio = AudioSegment.from_file(str(input_path))
    # Set to 1 channel and 16kHz
    audio = audio.set_frame_rate(16000).set_channels(1)
    # Export as WAV
    audio.export(str(output_path), format="wav")

def format_time(seconds: float) -> float:
    """Format seconds to a float with two decimal places."""
    return round(seconds, 2)


def process_task(body: bytes) -> dict:
    """
    Process a single diarization task.
    """
    payload = json.loads(body)
    s3_key = payload["s3Key"]
    meeting_id = payload["meetingId"]
    logger.info(f"Processing meetingId={meeting_id}, s3_key={s3_key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        orig = tmpdir_path / f"{uuid.uuid4()}"
        wav = tmpdir_path / "converted.wav"

        # Download from S3
        s3.download_file(S3_BUCKET, s3_key, str(orig))
        # Convert to WAV
        convert_to_wav(orig, wav)
        # Run diarization
        diarization = pipeline(str(wav))
        data = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            data.append({
                "startTime": format_time(turn.start),
                "endTime": format_time(turn.end),
                "speaker": speaker
            })

    return {
        "meetingId": meeting_id,
        "type": "SPEAKER",
        "data": data
    }


def consumer():
    """
    RabbitMQ consumer that processes diarization tasks.
    """
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
    )
    channel = connection.channel()

    # # Declare request queue
    # channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    # # Declare response exchange
    # channel.exchange_declare(exchange=RESPONSE_EXCHANGE, exchange_type='direct', durable=True)
    channel.basic_qos(prefetch_count=1)

    def on_message(ch, method, properties, body):
        try:
            result = process_task(body)
            # Publish to exchange
            ch.basic_publish(
                exchange=MQ_VOICE_EXCHANGE,
                routing_key=MQ_VOICE_RESULT_ROUTING_KEY,
                body=json.dumps(result),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Finished meetingId={result['meetingId']}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=MQ_VOICE_SPEAKER, on_message_callback=on_message)
    logger.info("Consumer started, waiting for messages...")
    channel.start_consuming()

def start_consumer_thread():
    """
    Start the RabbitMQ consumer in a background thread.
    """
    thread = Thread(target=consumer, daemon=True)
    thread.start()
