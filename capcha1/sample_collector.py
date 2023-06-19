import time

import requests

request_url = "https://defendtheweb.net/extras/playground/captcha/captcha1.php"
sample_count = 200
offset = 0
sample_output = "./samples"
remaining_samples = sample_count
estimated_time = remaining_samples * 0.5
original_estimated_time = estimated_time


# Recalculate estimated time, based on remaining samples and average download time
def recalculate_estimated_time(remaining_samples):
    global estimated_time
    estimated_time = remaining_samples * 0.5


print("Downloading " + str(sample_count) + " samples from " + request_url)
# Download samples
for i in range(offset, sample_count):
    recalculate_estimated_time(remaining_samples)
    r = requests.get(request_url)
    with open(sample_output + "/" + str(i) + ".png", "wb") as f:
        f.write(r.content)
    print("Downloaded sample " + str(i) + "(" + str(estimated_time) + " / " + str(original_estimated_time) + ")")
    time.sleep(0.5)
    remaining_samples -= 1
