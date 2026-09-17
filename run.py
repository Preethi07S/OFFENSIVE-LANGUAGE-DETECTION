from pipeline import run_pipeline

result = run_pipeline(
    "test_video.mp4",
    work_dir="workdir"
)

print("\n===== TRANSCRIPT =====")
print(result["transcript"])

print("\n===== OFFENSIVE WORDS =====")
print(result["toxic_words"])

print("\n===== TOXIC TIMESTAMPS =====")
print(result["toxic_timestamps"])

print("\n===== OUTPUT VIDEO =====")
print(result["censored_video_path"])