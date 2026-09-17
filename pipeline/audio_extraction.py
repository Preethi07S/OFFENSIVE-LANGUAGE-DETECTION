"""Extract the audio track from a video file."""
from moviepy.editor import VideoFileClip


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract the audio track from ``video_path`` and save it as a WAV file.

    Args:
        video_path: Path to the source video file.
        audio_path: Path to write the extracted audio (.wav).

    Returns:
        The path to the extracted audio file.
    """
    video = VideoFileClip(video_path)
    try:
        video.audio.write_audiofile(audio_path, logger=None)
    finally:
        video.close()
    return audio_path
