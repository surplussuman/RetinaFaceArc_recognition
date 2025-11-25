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
            max_value=5,
            value=1,
            help="Skip frames for faster processing (1 = all frames)"
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
    tabs = st.tabs(["📹 Video Recognition", "➕ Enroll Users", "📊 About"])
    
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
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames", f"{total_frames:,}")
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("Resolution", f"{width}×{height}")
            col4.metric("Duration", format_duration(duration))
            
            # Process button
            if st.button("🚀 Start Recognition", type="primary", disabled=st.session_state.processing):
                st.session_state.processing = True
                
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
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                # Callbacks for live updates
                def update_progress(frame_idx, total_frames):
                    progress = int((frame_idx / total_frames) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_idx}/{total_frames}...")
                
                def update_preview(annotated_frame, frame_stats):
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(rgb_frame, caption=f"Frame {frame_stats['frame']} - Detections: {frame_stats['detections']}, Tracks: {frame_stats['tracks']}", width="stretch")
                
                # Process video
                status_text.text("Processing video...")
                
                try:
                    stats = recognizer.process_video(
                        video_path,
                        output_path=output_video,
                        show_preview=False,
                        max_frames=max_frames if max_frames > 0 else None,
                        progress_callback=update_progress,
                        frame_callback=update_preview if show_preview else None
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Processing complete!")
                    
                    # Save recognition log
                    if stats['recognition_log']:
                        df = pd.DataFrame(stats['recognition_log'])
                        df.to_csv(output_log, index=False)
                    
                    # Display results
                    st.success("✅ Video processed successfully!")
                    
                    # Statistics
                    st.subheader("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Detections", stats['total_detections'])
                    col2.metric("Unique Tracks", stats['total_tracks'])
                    col3.metric("Recognized", stats['total_recognized'])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg FPS", f"{stats['avg_fps']:.1f}")
                    col2.metric("Processing Time", format_duration(stats['processing_time']))
                    col3.metric("Unique Identities", len(stats.get('unique_identities', [])))
                    
                    # Identified people
                    if stats.get('unique_identities'):
                        st.info(f"**Identified:** {', '.join(stats['unique_identities'])}")
                    
                    # Recognition log
                    if stats['recognition_log']:
                        st.subheader("🔍 Recognition Log")
                        df = pd.DataFrame(stats['recognition_log'])
                        st.dataframe(df, use_container_width=True)
                    
                    # Download buttons
                    st.subheader("💾 Downloads")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
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
                    
                    # Video player
                    st.subheader("🎬 Preview")
                    st.video(str(output_video))
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                finally:
                    st.session_state.processing = False
                    # Cleanup
                    if video_path.exists():
                        video_path.unlink()
    
    # Tab 2: Enrollment
    with tabs[1]:
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
            with st.spinner(f"Enrolling {user_id}..."):
                try:
                    # Save photos to temp directory
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    for i, photo in enumerate(uploaded_photos):
                        photo_path = temp_dir / f"{i}_{photo.name}"
                        with open(photo_path, 'wb') as f:
                            f.write(photo.read())
                    
                    # Enroll
                    db_path = project_root / 'data' / 'face_database'
                    
                    enrollment = MultiQualityEnroller()
                    result = enrollment.enroll_from_directory(
                        user_id=user_id,
                        photos_directory=temp_dir,
                        db_path=db_path
                    )
                    
                    # Cleanup
                    shutil.rmtree(temp_dir)
                    
                    # Reload database
                    st.session_state.database = load_database()
                    
                    # Show results
                    st.success(f"✅ Enrolled {user_id} successfully!")
                    st.json({
                        'photos_processed': result['photos_processed'],
                        'total_embeddings': result['total_embeddings'],
                        'embeddings_per_photo': result['embeddings_per_photo']
                    })
                    
                except Exception as e:
                    st.error(f"❌ Enrollment failed: {str(e)}")
    
    # Tab 3: About
    with tabs[2]:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Features
        
        - **Multi-Quality Enrollment**: Generates 5 quality variants per photo for robust recognition
        - **Temporal Smoothing**: Averages embeddings across frames for stable identification
        - **Track-by-Detection**: IOU-based tracking maintains consistent identities
        - **CCTV Optimization**: Handles low resolution, compression artifacts, and motion blur
        - **Adaptive Thresholding**: Automatically adjusts recognition threshold based on image quality
        
        ### 🔬 Technical Details
        
        - **Detection**: MTCNN face detector
        - **Embedding**: ArcFace (ResNet-100, 512-dim)
        - **Database**: FAISS for efficient similarity search
        - **Tracking**: Exponential moving average (α=0.3) for identity voting
        - **Recognition Threshold**: 0.4 (adjustable based on accuracy requirements)
        
        ### 📈 Performance
        
        - **Image Recognition**: 75% on 40+ year old photos
        - **Multi-Person Support**: Tracks multiple faces simultaneously
        - **Real-Time Capable**: Configurable frame skip for performance tuning
        
        ### 📚 Documentation
        
        See `TECHNICAL_DOCUMENTATION.md` for detailed mathematical foundations and performance analysis.
        """)


if __name__ == '__main__':
    main()
