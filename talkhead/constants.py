import os
import dotenv
dotenv.load_dotenv()

TALKHEAD_SERVER_HOST = os.getenv("TALKHEAD_SERVER_HOST", "http://localhost:8000")
SCORING_SERVER_ENDPOINT=os.getenv("SCORING_SERVER_ENDPOINT", "http://localhost:8100/score")
BURN_UID = int(os.getenv("BURN_UID", 0))
BURN_RATIO = float(os.getenv("BURN_RATIO", 1.0))

MINER_SERVER_ENDPOINT=os.getenv("MINER_SERVER_ENDPOINT", "http://localhost:9000/generate")

DENDRITE_TIMEOUT = 120