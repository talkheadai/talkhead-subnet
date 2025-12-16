import os
import dotenv
dotenv.load_dotenv()

TALKHEAD_SERVER_ENDPOINT = os.getenv("TALKHEAD_SERVER_ENDPOINT", "http://localhost:8000/challenge")
SCORING_SERVER_ENDPOINT=os.getenv("SCORING_SERVER_ENDPOINT", "http://localhost:8100/score")
MINER_SERVER_ENDPOINT=os.getenv("MINER_SERVER_ENDPOINT", "http://localhost:9000/generate")

DENDRITE_TIMEOUT = 120