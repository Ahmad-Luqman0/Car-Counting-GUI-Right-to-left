This is a PyQt5-based desktop application for detecting vehicles (cars and motorcycles) entering a site from surveillance videos. 
It uses YOLOv8 object detection with ByteTrack tracking to count vehicles crossing a user-defined entry line. 
The app supports multiple video files, real-time preview, adjustable counters, logging, and CSV report generation.



Features
	•	Video Input
	  •	 Upload one or more videos, or select a folder containing videos.
	•	Log Folder & Metadata
	  •	Prompt for Site Name and Date, used to organize logs and reports.
	•	Interactive Entry Line Drawing
	  •	Click two points on the first frame to draw an entry line.
	•	Vehicle Detection & Tracking
	  •	Powered by YOLOv8 + ByteTrack for consistent object IDs.
	  •	Detects Cars and Motorcycles (with heuristics to avoid bicycles).
	• Counting Logic
  	•	Counts only when a vehicle crosses the entry line (right → left).
  	•	One screenshot per crossing is saved.
  •	GUI Controls
  	•	Start, Pause, Resume, Stop detection.
  	•	Manual adjustment of counts (+/−).
	•	Real-time Preview
	  •	Bounding boxes, labels, entry line, and counters displayed.
	•	Logging & Reports
  	• Per-video log: CSV with TrackID, timestamp, class, cropped image path.
  	•	Per-video summary inside log file.
  	•	Global summary: total_summary.csv with site name, date, video ID, start/end time, and counts.

   How It Works
	1.	Detection → YOLO detects objects per frame.
	2.	Tracking → ByteTrack assigns consistent IDs.
	3.	Crossing Check → Vehicle is counted only when crossing the entry line (using sign of cross product).
	4.	Logging → Each crossing saves: TrackID, timestamp, class, cropped image.
	5.	Summaries → Auto-generated per video and across all videos.

 	Steps in GUI:
	•	Upload video(s) or select a folder.
	•	Choose log folder → enter Date + Site Name.
	•	Start detection → click two points on preview to draw entry line.
	•	Detection runs in real-time (with frame skipping for speed).
	•	Use Pause/Resume/Stop as needed.
	•	Adjust counters manually with +/− buttons.
