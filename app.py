"""
Face Recognition System - Streamlit Web Interface
=================================================

Web interface for video face recognition with CCTV support.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
import tempfile
import shutil
import pandas as pd
from datetime import datetime
from core.vector_db import VectorDatabase
from core.video_recognition import create_video_recognizer
from core.multi_quality_enrollment import MultiQualityEnroller
from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av as _av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
import cv2

# Page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Initialize session state
if 'database' not in st.session_state:
    st.session_state.database = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


def count_user_embeddings(vector_db: VectorDatabase, user_id: str) -> int:
    """Count total embeddings for a user."""
    count = 0
    for uid in vector_db.user_ids:
        if uid == user_id:
            count += 1
    return count


def load_database():
    """Load face database."""
    db_path = project_root / 'data' / 'face_database'
    index_path = Path(str(db_path) + '.index')
    
    if not index_path.exists():
        return None
    
    vector_db = VectorDatabase(embedding_dim=512)
    vector_db.load(db_path)
    return vector_db


def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    st.title("🎥 Face Recognition System")
    st.markdown("**CCTV-Grade Video Face Recognition with Temporal Tracking**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load database
        if st.button("🔄 Reload Database"):
            st.session_state.database = load_database()
        
        if st.session_state.database is None:
            st.session_state.database = load_database()
        
        # Database info
        if st.session_state.database:
            user_count = st.session_state.database.get_user_count()
            st.success(f"✓ Database loaded")
            st.info(f"**Enrolled Users:** {user_count}")
            
            users = st.session_state.database.get_all_users()
            with st.expander("View enrolled users"):
                for user in users:
                    count = count_user_embeddings(st.session_state.database, user['user_id'])
                    st.write(f"• **{user['user_id']}** ({count} embeddings)")
        else:
            st.warning("⚠️ No database found")
            st.info("Please enroll users first using:\n```\npython tools/enroll_multi_quality.py\n```")
        
        st.divider()
        
        # Recognition settings
        st.subheader("🎯 Recognition")
        threshold = st.slider(
            "Recognition Threshold",
            min_value=0.2,
            max_value=0.6,
            value=0.4,
            step=0.05,
            help="Lower = more lenient, Higher = stricter"
        )
        
        st.subheader("⚡ Performance")
        skip_frames = st.slider(
            "Process Every N Frames",
            min_value=1,
            max_value=10,
            value=3,
            help="Process every Nth frame — 3 = 3× faster, 5 = 5× faster"
        )
        
        show_preview = st.checkbox(
            "Show Live Preview",
            value=False,
            help="Display processing preview (slower)"
        )
        
        max_frames = st.number_input(
            "Max Frames (0 = all)",
            min_value=0,
            value=0,
            step=100,
            help="Limit frames for testing"
        )
    
    # Main content
    tabs = st.tabs(["📹 Video Recognition", "📡 Live Stream", "➕ Enroll Users", "📊 About"])
    
    # Tab 1: Video Recognition
    with tabs[0]:
        st.header("Video Recognition")
        
        if not st.session_state.database:
            st.error("⚠️ Please enroll users before processing videos")
            return
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = Path(tmp_file.name)
            
            # Video info
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get first frame for zone configuration
            ret, first_frame = cap.read()
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames", f"{total_frames:,}")
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("Resolution", f"{width}×{height}")
            col4.metric("Duration", format_duration(duration))
            
            # Zone Configuration Section
            st.divider()
            st.subheader("🎯 Detection Zone Configuration (Optional)")
            st.markdown("**Define zones where faces are expected to speed up processing**")
            
            enable_zones = st.checkbox("Enable Zone-Based Detection", value=False, 
                                      help="Process only specific regions (doors, entry points) for faster detection")
            
            if enable_zones:
                # Initialize session state for zones
                if 'zones' not in st.session_state:
                    st.session_state.zones = []
                
                st.info("💡 **Tip:** Define zones at entry points, checkout counters, or doorways where faces appear")
                
                # Show preview frame
                if ret and first_frame is not None:
                    col_left, col_right = st.columns([2, 1])
                    
                    with col_left:
                        st.markdown("**Preview Frame with Zones:**")
                        
                        # Draw zones on preview
                        preview_frame = first_frame.copy()
                        for i, zone in enumerate(st.session_state.zones):
                            if zone.get('enabled', True):
                                x, y, w, h = zone['roi']
                                cv2.rectangle(preview_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                label = f"{zone['name']}"
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                cv2.rectangle(preview_frame, (x, y-label_h-15), (x+label_w+10, y), (0, 255, 0), -1)
                                cv2.putText(preview_frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), 
                                caption="Green boxes show detection zones", use_container_width=True)
                    
                    with col_right:
                        st.markdown("**Zone Controls:**")
                        
                        zone_action = st.radio("Action:", ["Add Zone", "Edit Zone", "Delete Zone"], key="zone_action")
                        
                        if zone_action == "Add Zone":
                            with st.form("add_zone_form"):
                                zone_name = st.text_input("Zone Name", value=f"Zone {len(st.session_state.zones) + 1}")
                                
                                x = st.number_input("X (left)", 0, width, width//4, 10)
                                y = st.number_input("Y (top)", 0, height, height//4, 10)
                                w = st.number_input("Width", 100, width, width//2, 10)
                                h = st.number_input("Height", 100, height, height//2, 10)
                                
                                if st.form_submit_button("➕ Add Zone"):
                                    new_zone = {
                                        'name': zone_name,
                                        'roi': [int(x), int(y), int(w), int(h)],
                                        'enabled': True
                                    }
                                    st.session_state.zones.append(new_zone)
                                    st.success(f"✅ Added: {zone_name}")
                                    st.rerun()
                        
                        elif zone_action == "Edit Zone" and st.session_state.zones:
                            zone_idx = st.selectbox("Select Zone:", 
                                                   range(len(st.session_state.zones)),
                                                   format_func=lambda i: st.session_state.zones[i]['name'])
                            
                            zone = st.session_state.zones[zone_idx]
                            
                            with st.form("edit_zone_form"):
                                new_name = st.text_input("Zone Name", value=zone['name'])
                                x = st.number_input("X (left)", 0, width, zone['roi'][0], 10)
                                y = st.number_input("Y (top)", 0, height, zone['roi'][1], 10)
                                w = st.number_input("Width", 100, width, zone['roi'][2], 10)
                                h = st.number_input("Height", 100, height, zone['roi'][3], 10)
                                enabled = st.checkbox("Enabled", value=zone.get('enabled', True))
                                
                                if st.form_submit_button("💾 Save"):
                                    st.session_state.zones[zone_idx] = {
                                        'name': new_name,
                                        'roi': [int(x), int(y), int(w), int(h)],
                                        'enabled': enabled
                                    }
                                    st.success(f"✅ Updated: {new_name}")
                                    st.rerun()
                        
                        elif zone_action == "Delete Zone" and st.session_state.zones:
                            zone_idx = st.selectbox("Select Zone:", 
                                                   range(len(st.session_state.zones)),
                                                   format_func=lambda i: st.session_state.zones[i]['name'])
                            
                            if st.button("🗑️ Delete", type="primary"):
                                deleted_name = st.session_state.zones[zone_idx]['name']
                                del st.session_state.zones[zone_idx]
                                st.success(f"🗑️ Deleted: {deleted_name}")
                                st.rerun()
                        
                        # Zone Statistics
                        if st.session_state.zones:
                            st.divider()
                            st.markdown("**📊 Zone Stats:**")
                            total_zone_pixels = sum(z['roi'][2] * z['roi'][3] for z in st.session_state.zones if z.get('enabled', True))
                            full_frame_pixels = width * height
                            reduction = (1 - total_zone_pixels / full_frame_pixels) * 100 if full_frame_pixels > 0 else 0
                            
                            st.metric("Active Zones", len([z for z in st.session_state.zones if z.get('enabled', True)]))
                            st.metric("Pixel Reduction", f"{reduction:.1f}%")
                            st.metric("Expected Speedup", f"{full_frame_pixels / total_zone_pixels if total_zone_pixels > 0 else 1:.1f}×")
            
            st.divider()
            
            # Process button
            if st.button("🚀 Start Recognition", type="primary", disabled=st.session_state.processing):
                st.session_state.processing = True
                
                # Save zones to config if enabled
                if enable_zones and st.session_state.zones:
                    import yaml
                    config_path = project_root / 'config' / 'system_config.yaml'
                    
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    if 'camera_zone' not in config:
                        config['camera_zone'] = {}
                    
                    config['camera_zone']['enabled'] = True
                    config['camera_zone']['zones'] = st.session_state.zones
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    
                    st.info(f"✅ Using {len(st.session_state.zones)} detection zone(s)")
                
                # Create recognizer
                recognizer = create_video_recognizer(
                    st.session_state.database,
                    recognition_threshold=threshold,
                    process_every_n_frames=skip_frames
                )
                
                # Create output paths
                output_dir = project_root / 'results' / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_video = output_dir / f"output_{uploaded_file.name}"
                output_log = output_dir / "recognition_log.csv"
                
                # Progress bar — throttled to 1 update per 25 frames to avoid
                # WebSocket round-trip overhead (each st.* call ≈ 30-50ms)
                progress_bar = st.progress(0)
                status_text = st.empty()
                _last_progress_update = [0]
                
                def update_progress(frame_idx, total_frames):
                    # Only update Streamlit UI every 25 frames to avoid per-frame overhead
                    if frame_idx - _last_progress_update[0] < 25 and frame_idx < total_frames:
                        return
                    _last_progress_update[0] = frame_idx
                    progress = min(int((frame_idx / total_frames) * 100), 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing… {frame_idx}/{total_frames} frames ({progress}%)")
                
                status_text.text("Processing video (offline — results shown when done) …")
                
                try:
                    # If the user enabled Show Live Preview in the sidebar, provide
                    # a throttled Streamlit `frame_callback` to display annotated
                    # frames. Do NOT enable `show_preview` (cv2.imshow) because
                    # that opens a native window on the server. The frame callback
                    # is lighter and avoids per-frame UI storms by throttling.
                    preview_placeholder = st.empty()
                    _last_preview_update = [0]

                    def _frame_callback(annotated_frame, stats_cb):
                        # Throttle preview updates to avoid UI overload (every 5 frames)
                        frame_idx_cb = stats_cb.get('frame', 0)
                        if frame_idx_cb - _last_preview_update[0] < 5:
                            return
                        _last_preview_update[0] = frame_idx_cb
                        try:
                            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            # `use_column_width` is deprecated; use `width` instead.
                            preview_placeholder.image(rgb, width=640)
                        except Exception:
                            # Don't fail processing if preview drawing errors
                            pass

                    frame_cb = _frame_callback if show_preview else None

                    stats = recognizer.process_video(
                        video_path,
                        output_path=output_video,
                        show_preview=False,
                        max_frames=max_frames if max_frames > 0 else None,
                        progress_callback=update_progress,
                        frame_callback=frame_cb,
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Processing complete!")
                    
                    # Save recognition log
                    recognition_log = stats.get('recognition_log', [])
                    if recognition_log:
                        df_log = pd.DataFrame(recognition_log)
                        df_log.to_csv(output_log, index=False)
                    
                    st.success("✅ Video processed successfully!")
                    
                    # --- Statistics ---
                    st.subheader("📊 Results")
                    
                    total_detections = stats.get('total_detections', 0)
                    unique_tracks = stats.get('unique_tracks', 0)
                    recognized_ids = stats.get('recognized_identities', [])
                    avg_ms = stats.get('avg_processing_time', 0)
                    total_frames = stats.get('total_frames', 1)
                    processed_frames = stats.get('processed_frames', 0)
                    import time as _time
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Detections", total_detections)
                    col2.metric("Unique Tracks", unique_tracks)
                    col3.metric("Identified People", len(recognized_ids))
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg ms/frame", f"{avg_ms:.1f}")
                    col2.metric("Frames Processed", processed_frames)
                    col3.metric("Frames Total", total_frames)
                    
                    if recognized_ids:
                        names = [n for n in recognized_ids if n and n != "Unknown"]
                        if names:
                            st.info(f"**Identified:** {', '.join(sorted(set(names)))}")
                    
                    # --- Event Feed (detected face snapshots) ---
                    if recognition_log:
                        st.subheader("👤 Detected Faces (Event Feed)")
                        st.caption("Showing first occurrence of each unique identity.")
                        
                        seen_ids = set()
                        event_faces = []
                        cap_ev = cv2.VideoCapture(str(output_video))
                        for entry in recognition_log:
                            identity = entry.get('identity', 'Unknown')
                            if identity in seen_ids or identity == 'Unknown':
                                continue
                            seen_ids.add(identity)
                            # Seek to frame and grab crop
                            cap_ev.set(cv2.CAP_PROP_POS_FRAMES, entry['frame'])
                            ret_ev, fr_ev = cap_ev.read()
                            event_faces.append({
                                'identity': identity,
                                'confidence': entry.get('confidence', 0),
                                'timestamp': entry.get('timestamp', 0),
                                'frame': entry['frame'],
                                'image': fr_ev if ret_ev else None,
                            })
                        cap_ev.release()
                        
                        if event_faces:
                            cols = st.columns(min(len(event_faces), 4))
                            for idx, ev in enumerate(event_faces):
                                with cols[idx % 4]:
                                    if ev['image'] is not None:
                                        rgb = cv2.cvtColor(ev['image'], cv2.COLOR_BGR2RGB)
                                        st.image(rgb, caption=f"{ev['identity']} ({ev['confidence']:.2f})\n@ {ev['timestamp']:.1f}s", use_container_width=True)
                                    else:
                                        st.write(f"**{ev['identity']}** @ {ev['timestamp']:.1f}s")
                    
                    # --- Full log table ---
                    if recognition_log:
                        st.subheader("🔍 Recognition Log")
                        df = pd.DataFrame(recognition_log)
                        st.dataframe(df, use_container_width=True)
                    
                    # --- Downloads ---
                    st.subheader("💾 Downloads")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if output_video.exists():
                            with open(output_video, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Annotated Video",
                                    data=f,
                                    file_name=output_video.name,
                                    mime="video/mp4"
                                )
                    
                    with col2:
                        if output_log.exists():
                            with open(output_log, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Recognition Log",
                                    data=f,
                                    file_name="recognition_log.csv",
                                    mime="text/csv"
                                )
                    
                    # --- Video playback (after processing) ---
                    if output_video.exists():
                        st.subheader("🎬 Annotated Video")
                        st.video(str(output_video))
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.code(traceback.format_exc())
                finally:
                    st.session_state.processing = False
                    if video_path.exists():
                        try:
                            video_path.unlink()
                        except Exception:
                            pass
    
    # Tab 2: Live Stream
    with tabs[1]:
        st.header("Live Face Recognition Stream")
        
        if not WEBRTC_AVAILABLE:
            st.error("streamlit-webrtc not installed. Run: pip install streamlit-webrtc av")
        elif not st.session_state.database:
            st.error("⚠️ Please enroll users first (Enroll Users tab)")
        else:
            st.info("📡 Webcam stream with real-time face recognition. Faces are detected and identified live.")
            
            col_left, col_right = st.columns([3, 1])
            with col_right:
                live_threshold = st.slider("Recognition Threshold", 0.2, 0.6, 0.4, 0.05, key="live_thresh")
                live_skip = st.slider("Process Every N Frames", 1, 10, 3, key="live_skip")
            
            # Build recognizer reference shared across WebRTC frames
            if 'live_recognizer' not in st.session_state or st.session_state.live_recognizer is None:
                st.session_state.live_recognizer = create_video_recognizer(
                    st.session_state.database,
                    recognition_threshold=live_threshold,
                    process_every_n_frames=live_skip,
                )

            # Capture recognizer in local variable and pass into the processor.
            # Avoid accessing `st.session_state` from the processor thread.
            recognizer_ref = st.session_state.live_recognizer

            # VideoProcessor runs in its own thread managed by streamlit-webrtc
            class _LiveProcessor(VideoProcessorBase):
                def __init__(self, recognizer):
                    self.recognizer = recognizer
                    self.frame_count = 0

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    result = self.recognizer.process_frame(img, self.frame_count)
                    self.frame_count += 1
                    if not result.get('skipped') and not result.get('motion_gated'):
                        img = self.recognizer.draw_annotations(img, result)
                    return _av.VideoFrame.from_ndarray(img, format="bgr24")

            with col_left:
                webrtc_streamer(
                    key="face-recognition-live",
                    video_processor_factory=lambda: _LiveProcessor(recognizer_ref),
                    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            
            if st.button("🔄 Reset Live Stream Recognizer"):
                st.session_state.live_recognizer = None
                st.rerun()
    
    # Tab 3: Enrollment
    with tabs[2]:
        st.header("Enroll New Users")
        
        st.info("""
        **Multi-Quality Enrollment** generates multiple quality variants of each photo 
        to improve recognition accuracy on low-quality CCTV footage.
        
        For best results:
        - Upload 3-5 clear photos of the person
        - Photos should have different angles and expressions
        - Ensure good lighting and focus
        """)
        
        user_id = st.text_input("User ID/Name", placeholder="e.g., John_Doe")
        
        uploaded_photos = st.file_uploader(
            "Upload Photos",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload 3-5 photos for best results"
        )
        
        if st.button("📝 Enroll User", type="primary") and user_id and uploaded_photos:
            with st.spinner(f"Enrolling {user_id}... (this may take 30-60s)"):
                try:
                    # Save photos to temp directory
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    for i, photo in enumerate(uploaded_photos):
                        photo_path = temp_dir / f"{i}_{photo.name}"
                        with open(photo_path, 'wb') as f:
                            f.write(photo.read())
                    
                    # Use the loaded database from session state
                    vector_db = st.session_state.database
                    if vector_db is None:
                        vector_db = VectorDatabase(embedding_dim=512)
                        db_path = project_root / 'data' / 'face_database'
                        index_path = Path(str(db_path) + '.index')
                        if index_path.exists():
                            vector_db.load(db_path)
                    
                    # Initialize all required components
                    detector = RetinaFaceDetector()
                    aligner = FaceAligner()
                    embedder = ArcFaceEmbedder()
                    
                    enrollment = MultiQualityEnroller(detector, aligner, embedder, vector_db)
                    result = enrollment.enroll_from_directory(
                        user_id=user_id,
                        directory=temp_dir,
                        replace_existing=True,
                    )
                    
                    # Cleanup temp files
                    shutil.rmtree(temp_dir)
                    
                    if result.get('success'):
                        # Persist the updated database
                        db_path = project_root / 'data' / 'face_database'
                        vector_db.save(db_path)
                        
                        # Reload database in session state
                        st.session_state.database = vector_db
                        
                        st.success(f"✅ Enrolled {user_id} successfully!")
                        st.json({
                            'photos_processed': result.get('num_images', 0),
                            'total_embeddings': result.get('num_embeddings', 0),
                        })
                    else:
                        st.error(f"❌ Enrollment failed: {result.get('reason', 'unknown')}")
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ Enrollment failed: {str(e)}")
                    st.code(traceback.format_exc())
    
    # Tab 4: About
    with tabs[3]:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Features
        
        - **Multi-Quality Enrollment**: Generates 5 quality variants per photo for robust recognition
        - **Temporal Smoothing**: Averages embeddings across frames for stable identification
        - **Track-by-Detection**: IOU-based tracking maintains consistent identities
        - **CCTV Optimization**: Handles low resolution, compression artifacts, and motion blur
        - **Motion Gate**: Event-driven detection — detector runs only when motion detected
        - **Ghost Tracking**: Identity cached per track, embedding re-computed every 30 frames
        
        ### 🔬 Technical Details
        
        - **Detection**: RetinaFace ResNet50 (ONNX, CPU)
        - **Embedding**: ArcFace ResNet100 (ONNX, CPU, 512-dim)
        - **Database**: FAISS IndexFlatL2 with cosine similarity conversion
        - **Tracking**: IOU-based track assignment + EMA identity voting (α=0.3)
        - **Recognition Threshold**: 0.4 cosine similarity (adjustable)
        
        ### 📚 Documentation
        
        See `docs/TECHNICAL_DOCUMENTATION.md` for mathematical foundations.
        See `docs/EVENT_DRIVEN_ARCHITECTURE_ANALYSIS.md` for architecture analysis.
        """)


if __name__ == '__main__':
    main()

