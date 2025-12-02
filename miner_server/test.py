from main import generate, GenerateRequest
import base64
import time

if __name__ == "__main__":
    # test the generate function
    start_time = time.time()
    response = generate(GenerateRequest(
        text = "TalkHead Subnet is a Bittensor subnet focused on generating high-quality talking head avatars. ",
        # "image_url": "https://i.postimg.cc/1tj4SZbT/cute.jpg",
        # image_url = "https://i.postimg.cc/DyH90PGg/hat.jpg",
        image_url = "https://i.postimg.cc/kGFtf6Ys/talker.jpg"
    ))
    print(f"ok: {response.ok}, error_code: {response.error_code}, error_message: {response.error_message}")
    with open(f"../test_data/talker.mp4", "wb") as f:
        f.write(base64.b64decode(response.video_base64))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")