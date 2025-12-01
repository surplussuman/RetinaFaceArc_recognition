"""
Zone Configuration Interface
Interactive UI for users to draw and configure detection zones on their camera feed
"""

import streamlit as st
import cv2
import numpy as np
import yaml
from pathlib import Path
import json

st.set_page_config(
    page_title="Zone Configuration - SmartEntry",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Detection Zone Configuration")
st.markdown("**Draw detection zones on your camera feed to optimize face recognition**")

# Load current config
config_path = Path("config/system_config.yaml")

def load_config():
    """Load current system configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Save updated configuration"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def load_test_frame(video_path):
    """Load first frame from video for zone configuration"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

# Sidebar - Video Selection
st.sidebar.header("1️⃣ Select Video Source")

video_source = st.sidebar.radio(
    "Choose video source:",
    ["Upload Video", "Sample Videos", "Webcam (Live)"]
)

test_frame = None
video_path = None

if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi', 'mkv'])
    if uploaded_file:
        # Save temporarily
        temp_path = Path("temp_uploaded_video.mp4")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        video_path = str(temp_path)
        test_frame = load_test_frame(video_path)

elif video_source == "Sample Videos":
    sample_videos = list(Path("sampleImages").glob("*.mp4")) + list(Path("sampleImages").glob("*.mov"))
    if sample_videos:
        selected = st.sidebar.selectbox("Select sample video:", [v.name for v in sample_videos])
        video_path = str(Path("sampleImages") / selected)
        test_frame = load_test_frame(video_path)

elif video_source == "Webcam (Live)":
    st.sidebar.info("📸 Webcam preview will show below. Click 'Capture Frame' to freeze it for zone drawing.")
    if st.sidebar.button("📸 Capture Frame from Webcam"):
        cap = cv2.VideoCapture(0)
        ret, test_frame = cap.read()
        cap.release()
        if ret:
            st.session_state['captured_frame'] = test_frame
    
    if 'captured_frame' in st.session_state:
        test_frame = st.session_state['captured_frame']

# Main Area - Zone Configuration
if test_frame is not None:
    st.success(f"✅ Frame loaded! Resolution: {test_frame.shape[1]}×{test_frame.shape[0]}")
    
    # Load current zones
    config = load_config()
    current_zones = config.get('camera_zone', {}).get('zones', [])
    
    # Initialize session state for zones
    if 'zones' not in st.session_state:
        if current_zones:
            st.session_state['zones'] = current_zones
        else:
            st.session_state['zones'] = []
    
    # Zone drawing interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Draw Detection Zones")
        
        # Zone controls
        zone_action = st.radio(
            "Action:",
            ["➕ Add New Zone", "✏️ Edit Existing Zone", "🗑️ Delete Zone"],
            horizontal=True
        )
        
        if zone_action == "➕ Add New Zone":
            st.markdown("**Define new zone:**")
            
            zone_name = st.text_input("Zone Name:", value=f"Zone {len(st.session_state['zones']) + 1}")
            
            col_x, col_y, col_w, col_h = st.columns(4)
            
            with col_x:
                x = st.number_input("X (left)", min_value=0, max_value=test_frame.shape[1], value=test_frame.shape[1]//4, step=10)
            with col_y:
                y = st.number_input("Y (top)", min_value=0, max_value=test_frame.shape[0], value=test_frame.shape[0]//4, step=10)
            with col_w:
                w = st.number_input("Width", min_value=100, max_value=test_frame.shape[1], value=test_frame.shape[1]//2, step=10)
            with col_h:
                h = st.number_input("Height", min_value=100, max_value=test_frame.shape[0], value=test_frame.shape[0]//2, step=10)
            
            if st.button("➕ Add This Zone"):
                new_zone = {
                    'name': zone_name,
                    'roi': [int(x), int(y), int(w), int(h)],
                    'enabled': True
                }
                st.session_state['zones'].append(new_zone)
                st.success(f"✅ Added zone: {zone_name}")
                st.rerun()
        
        elif zone_action == "✏️ Edit Existing Zone":
            if st.session_state['zones']:
                zone_idx = st.selectbox(
                    "Select zone to edit:",
                    range(len(st.session_state['zones'])),
                    format_func=lambda i: st.session_state['zones'][i]['name']
                )
                
                zone = st.session_state['zones'][zone_idx]
                
                st.markdown(f"**Editing: {zone['name']}**")
                
                new_name = st.text_input("Zone Name:", value=zone['name'])
                
                col_x, col_y, col_w, col_h = st.columns(4)
                
                with col_x:
                    x = st.number_input("X (left)", min_value=0, max_value=test_frame.shape[1], value=zone['roi'][0], step=10)
                with col_y:
                    y = st.number_input("Y (top)", min_value=0, max_value=test_frame.shape[0], value=zone['roi'][1], step=10)
                with col_w:
                    w = st.number_input("Width", min_value=100, max_value=test_frame.shape[1], value=zone['roi'][2], step=10)
                with col_h:
                    h = st.number_input("Height", min_value=100, max_value=test_frame.shape[0], value=zone['roi'][3], step=10)
                
                enabled = st.checkbox("Zone Enabled", value=zone.get('enabled', True))
                
                if st.button("💾 Save Changes"):
                    st.session_state['zones'][zone_idx] = {
                        'name': new_name,
                        'roi': [int(x), int(y), int(w), int(h)],
                        'enabled': enabled
                    }
                    st.success(f"✅ Updated zone: {new_name}")
                    st.rerun()
            else:
                st.info("No zones to edit. Add a zone first!")
        
        elif zone_action == "🗑️ Delete Zone":
            if st.session_state['zones']:
                zone_idx = st.selectbox(
                    "Select zone to delete:",
                    range(len(st.session_state['zones'])),
                    format_func=lambda i: st.session_state['zones'][i]['name']
                )
                
                if st.button("🗑️ Delete This Zone", type="primary"):
                    deleted_name = st.session_state['zones'][zone_idx]['name']
                    del st.session_state['zones'][zone_idx]
                    st.success(f"🗑️ Deleted zone: {deleted_name}")
                    st.rerun()
            else:
                st.info("No zones to delete.")
        
        # Draw zones on frame
        preview_frame = test_frame.copy()
        
        for i, zone in enumerate(st.session_state['zones']):
            if zone.get('enabled', True):
                x, y, w, h = zone['roi']
                
                # Draw rectangle
                cv2.rectangle(preview_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Draw label
                label = f"{zone['name']}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(preview_frame, (x, y-label_h-15), (x+label_w+10, y), (0, 255, 0), -1)
                cv2.putText(preview_frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Draw zone info
                pixel_count = w * h
                info_text = f"{w}x{h} = {pixel_count:,} pixels"
                cv2.putText(preview_frame, info_text, (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display preview
        st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Zone Preview", use_container_width=True)
    
    with col2:
        st.subheader("📋 Current Zones")
        
        if st.session_state['zones']:
            for i, zone in enumerate(st.session_state['zones']):
                with st.expander(f"{'✅' if zone.get('enabled', True) else '❌'} {zone['name']}"):
                    st.write(f"**Position:** X={zone['roi'][0]}, Y={zone['roi'][1]}")
                    st.write(f"**Size:** {zone['roi'][2]}×{zone['roi'][3]}")
                    st.write(f"**Pixels:** {zone['roi'][2] * zone['roi'][3]:,}")
                    st.write(f"**Status:** {'🟢 Enabled' if zone.get('enabled', True) else '🔴 Disabled'}")
        else:
            st.info("No zones configured yet. Add a zone to get started!")
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state['zones']:
            total_zone_pixels = sum(z['roi'][2] * z['roi'][3] for z in st.session_state['zones'] if z.get('enabled', True))
            full_frame_pixels = test_frame.shape[1] * test_frame.shape[0]
            reduction = (1 - total_zone_pixels / full_frame_pixels) * 100 if full_frame_pixels > 0 else 0
            speedup = full_frame_pixels / total_zone_pixels if total_zone_pixels > 0 else 1
            
            st.metric("Active Zones", len([z for z in st.session_state['zones'] if z.get('enabled', True)]))
            st.metric("Pixel Reduction", f"{reduction:.1f}%")
            st.metric("Expected Speedup", f"{speedup:.1f}×")
        else:
            st.metric("Active Zones", 0)
        
        st.divider()
        
        # Save configuration
        st.subheader("💾 Save Configuration")
        
        if st.button("💾 Save Zones to Config", type="primary"):
            config = load_config()
            
            # Update camera_zone section
            if 'camera_zone' not in config:
                config['camera_zone'] = {}
            
            config['camera_zone']['enabled'] = len(st.session_state['zones']) > 0
            config['camera_zone']['zones'] = st.session_state['zones']
            
            save_config(config)
            st.success("✅ Configuration saved!")
            st.balloons()
        
        if st.session_state['zones'] and video_path:
            st.divider()
            st.subheader("🚀 Start Recognition")
            
            if st.button("▶️ Process Video with Zones", type="primary"):
                st.info("Processing video... This will take a few moments.")
                st.write("Run this command in terminal:")
                st.code(f'python tools/process_video.py --video "{video_path}" --output "results/zone_output.mp4"')

else:
    st.info("👆 Please select a video source from the sidebar to get started.")
    
    st.markdown("""
    ### 📖 How to Use:
    
    1. **Select Video Source**: Choose from upload, sample videos, or webcam
    2. **Add Zones**: Define detection zones where faces are expected (doors, gates, checkout counters)
    3. **Adjust Position**: Use X, Y, Width, Height controls to position zones
    4. **Preview**: See green rectangles showing your zones
    5. **Save Configuration**: Click "Save Zones to Config" when ready
    6. **Process Video**: Run face recognition with optimized zone detection!
    
    ### 💡 Tips:
    
    - **Smaller zones = Faster processing**: Focus on entry/exit points
    - **Multiple zones**: Add separate zones for multiple doors/lanes
    - **Pixel reduction**: Aim for 80-95% reduction for best speed
    - **Zone size**: Keep zones 640×640 or smaller for optimal detector performance
    
    ### 🎯 Expected Performance:
    
    - **Full Frame (1920×1080)**: ~2-3 FPS on CPU
    - **Single Zone (640×640)**: ~8-12 FPS on CPU  
    - **Speedup**: 3-5× faster with proper zones!
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>SmartEntry Zone Configuration | Geometry Optimization for Real-Time Face Recognition</small>
</div>
""", unsafe_allow_html=True)
