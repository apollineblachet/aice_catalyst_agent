We want to build a web-based feature that automatically summarizes recorded team meetings.
The tool should accept uploaded audio or video files (MP4, WAV, MP3), transcribe them into text using an external API (e.g., OpenAI Whisper or AssemblyAI), and generate concise summaries highlighting key decisions, action items, and open questions.

Users should be able to:
- Upload a file via a simple UI form.
- View the transcript and summary in a dashboard.
- Download both as text or PDF.

Constraints:
- Backend: Python (FastAPI).
- Frontend: React, responsive, minimal UI.
- 1-hour meeting transcription ≤ 5 minutes.
- Summarization ≤ 30 seconds.

Success Criteria:
- Transcription accuracy ≥ 90%.
- Summaries capture ≥ 80% of key action items (human-validated).
- Support 10 concurrent uploads without performance degradation.