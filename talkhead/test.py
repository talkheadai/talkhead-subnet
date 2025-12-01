from talkhead.utils.r2 import upload_video_base64_to_r2
import base64
import time

filename = "test_data/result.mp4"
with open(filename, "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("ascii")
start_time = time.time()
video_url = upload_video_base64_to_r2(video_b64, prompt="TalkHead Subnet is a Bittensor subnet")
if video_url:
    print(f"🟢 Uploaded video to R2: {video_url} in {time.time() - start_time} seconds")
else:
    print("🟠 Unable to upload video to R2; video_url not set.")