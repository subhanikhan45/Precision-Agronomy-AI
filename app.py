import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import sqlite3
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
import os
import requests
import statistics

# --- 1. PAGE SETUP & UI POLISH ---
st.set_page_config(page_title="AI Precision Agronomy Pro", layout="wide")

# --- CUSTOM CSS UI POLISH (HOVER EFFECTS & CARDS) ---
st.markdown("""
<style>
    /* Add smooth hover zoom and shadows to all images */
    .stImage > img {
        border-radius: 15px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stImage > img:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0,204,102,0.4);
    }
    
    /* Style the Metric Cards to look like a modern dashboard */
    div[data-testid="metric-container"] {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00cc66;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE INIT ---
def init_db():
    conn = sqlite3.connect('farm_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (timestamp TEXT, crop TEXT, confidence REAL, severity REAL, dosage REAL, saved_rupees REAL, scan_type TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_scan_to_db(crop, conf, sev, dosage, saved_rupees, scan_type="Single"):
    conn = sqlite3.connect('farm_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO scans VALUES (?,?,?,?,?,?,?)", (timestamp, crop, conf, sev, dosage, saved_rupees, scan_type))
    conn.commit()
    conn.close()

# --- 2. LOAD THE AI BRAIN ---
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('leaf_ai_model.h5')

model = load_ai_model()
CLASS_NAMES = ['Apple', 'Corn', 'Grape', 'Tomato']

# --- APP NAVIGATION ---
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to:", ["📡 AI Field Scanner", "🗄️ Farm Analytics & History"])
st.sidebar.markdown("---")

# ==========================================
#        PAGE 1: THE AI SCANNER
# ==========================================
if app_mode == "📡 AI Field Scanner":
    st.title("🚁 AI Precision Spraying")
    st.markdown("Process single leaves or bulk drone image captures for field-scale agronomy.")

# --- LIVE WEATHER & REVERSE GEOCODING ENGINE ---
    st.sidebar.header("🌤️ Live Environmental Data")
    
    # 1. TRUE SATELLITE GPS ACQUISITION
    st.sidebar.info("🛰️ Awaiting GPS Signal...")
    gps_data = streamlit_geolocation()

    # Check if the user clicked the button and got coordinates
    if gps_data and gps_data.get('latitude') is not None:
        auto_lat = float(gps_data['latitude'])
        auto_lon = float(gps_data['longitude'])
        st.sidebar.success("GPS Lock Acquired!")
    else:
        # Failsafe: Default to Vijayawada if they haven't clicked the button yet
        auto_lat = 16.5062
        auto_lon = 80.6480 

    farm_lat = st.sidebar.number_input("Farm Latitude", value=auto_lat, format="%.4f")
    farm_lon = st.sidebar.number_input("Farm Longitude", value=auto_lon, format="%.4f")

    # 2. THE TRANSLATOR
    @st.cache_data(ttl=3600) 
    def get_location_name(lat, lon):
        try:
            headers = {'User-Agent': 'AI-Agronomy-Pro-VJA/1.0'}
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
            response = requests.get(url, headers=headers).json()
            
            address = response.get('address', {})
            city = address.get('city', address.get('town', address.get('village', address.get('state_district', 'Unknown Location'))))
            state = address.get('state', 'Andhra Pradesh')
            
            if not city:
                city = "Vijayawada"
                
            return f"{city}, {state}"
        except:
            return "Vijayawada, Andhra Pradesh"

    @st.cache_data(ttl=3600) 
    def get_live_weather(lat, lon):
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_probability_max&timezone=auto"
            response = requests.get(url).json()
            rain_prob = response['daily']['precipitation_probability_max'][0]
            return rain_prob
        except:
            return 0 

    location_name = get_location_name(farm_lat, farm_lon)
    st.sidebar.info(f"📍 **Detected Location:**\n\n{location_name}")

    rain_probability = get_live_weather(farm_lat, farm_lon)
    safe_to_spray = rain_probability < 50 

    if safe_to_spray:
        st.sidebar.success(f"☀️ **Clear Skies.** Rain Probability: {rain_probability}%. Safe for drone deployment.")
    else:
        st.sidebar.error(f"🌧️ **STORM WARNING.** Rain Probability: {rain_probability}%. Drone deployment restricted.")

    st.sidebar.markdown("---")

    # --- SIDEBAR ECONOMICS ENGINE ---
    st.sidebar.header("💰 Farm Economics Settings")
    field_size = st.sidebar.number_input("Field Size (Acres)", min_value=0.5, value=5.0, step=0.5)
    spray_vol = st.sidebar.number_input("Standard Spray (Liters/Acre)", min_value=1.0, value=10.0, step=1.0)
    chem_cost = st.sidebar.number_input("Pesticide Cost (₹/Liter)", min_value=10.0, value=500.0, step=50.0)

# --- HELPER FUNCTION: PDF GENERATOR ---
    def create_pdf_report(crop, num_images, sev, dosage, sav, rupees_saved, batch_rgb, batch_heat, batch_inf, trad_total_cost, safe_to_spray):
        pdf = FPDF()
        
        # --- PAGE 1: EXECUTIVE SUMMARY ---
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="AI Precision Agronomy - Drone Sweep Report", ln=True, align='C')
        pdf.line(10, 20, 200, 20)
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Diagnostic Data (Field Average):", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt=f"Images Processed in Sweep: {num_images} Scans", ln=True)
        pdf.cell(200, 10, txt=f"Dominant Crop Detected: {crop}", ln=True)
        pdf.cell(200, 10, txt=f"Average Field Infection: {sev:.2f}% of Surface Area", ln=True)
        pdf.cell(200, 10, txt=f"Variable-Rate Spray Dosage Required: {dosage:.2f}%", ln=True)
        pdf.cell(200, 10, txt=f"Eco-Savings (Chemical Volume): {sav:.2f}%", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"FINANCIAL ROI: Estimated Savings of Rs. {rupees_saved:,.2f}.", ln=True)
        pdf.ln(10)
        
        img_bgr_0 = cv2.cvtColor(np.array(batch_rgb[0]), cv2.COLOR_RGB2BGR)
        heat_bgr_0 = cv2.cvtColor(batch_heat[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite("temp_orig_main.jpg", img_bgr_0)
        cv2.imwrite("temp_heat_main.jpg", heat_bgr_0)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Visual Targeting Analysis (Representative Sample):", ln=True)
        pdf.ln(5)
        current_y = pdf.get_y()
        pdf.image("temp_orig_main.jpg", x=10, y=current_y, w=85)
        pdf.image("temp_heat_main.jpg", x=105, y=current_y, w=85)

        # --- NEW: MULTI-PAGE APPENDIX WITH ROI & UNIQUE FILES ---
        if num_images > 1:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Appendix: Sector-by-Sector Breakdown", ln=True, align='C')
            pdf.line(10, 20, 200, 20)
            pdf.ln(10)
            
            # Calculate base cost per sector
            sector_trad_cost = trad_total_cost / num_images

            for i in range(num_images):
                if pdf.get_y() > 180:
                    pdf.add_page()
                
                # Calculate ROI for this specific sector
                img_inf = batch_inf[i]
                if not safe_to_spray:
                    img_dosage = 0.0
                else:
                    img_dosage = 0.0 if img_inf < 1.0 else min(img_inf + 10.0, 100.0)
                
                sector_ai_cost = sector_trad_cost * (img_dosage / 100.0)
                sector_rupees_saved = sector_trad_cost - sector_ai_cost

                # Print the new text with the Savings included!
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(200, 8, txt=f"Sector {i+1}: {crop} | Infection: {img_inf:.1f}% | Sector Savings: Rs. {sector_rupees_saved:,.2f}", ln=True)
                pdf.ln(2)
                current_y = pdf.get_y()

                # THE FIX: Generate unique filenames using the 'i' index
                orig_name = f"temp_orig_{i}.jpg"
                heat_name = f"temp_heat_{i}.jpg"

                img_bgr = cv2.cvtColor(np.array(batch_rgb[i]), cv2.COLOR_RGB2BGR)
                heat_bgr = cv2.cvtColor(batch_heat[i], cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(orig_name, img_bgr)
                cv2.imwrite(heat_name, heat_bgr)

                pdf.image(orig_name, x=10, y=current_y, w=85)
                pdf.image(heat_name, x=105, y=current_y, w=85)

                pdf.set_y(current_y + 90)
        
        pdf.output("Farm_Report.pdf")
        with open("Farm_Report.pdf", "rb") as pdf_file:
            return pdf_file.read()

    # --- 3. UI: IMAGE ACQUISITION ---
    st.markdown("### 📸 Image Acquisition (Batch Upload Enabled)")
    tab1, tab2 = st.tabs(["📁 Upload Images (Drone Batch)", "📷 Live Camera Feed"])

    with tab1:
        uploaded_files = st.file_uploader("Select one or multiple leaf images (JPG/PNG) to simulate a drone sweep", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    with tab2:
        camera_photo = st.camera_input("Take a live picture of a single leaf")

    images_to_process = []
    if camera_photo is not None:
        images_to_process.append(camera_photo)
    elif uploaded_files:
        images_to_process.extend(uploaded_files)

    if images_to_process:
        batch_id = "".join([img.name for img in images_to_process]) + str(len(images_to_process))
        
        if 'last_batch_id' not in st.session_state or st.session_state.last_batch_id != batch_id:
            st.session_state.batch_crops = []
            st.session_state.batch_confidences = []
            st.session_state.batch_infections = []
            
            st.session_state.batch_images_rgb = []
            st.session_state.batch_masks = []
            st.session_state.batch_overlays = []
            
            progress_text = "AI is processing Drone Sweep Data..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, img_file in enumerate(images_to_process):
                image = Image.open(img_file)
                image_rgb = image.convert('RGB') 
                img_cv = np.array(image_rgb) 
                
                # 1. OpenCV Processing (Run this FIRST to check for Chlorophyll)
                img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
                
                lower_green = np.array([25, 40, 40])
                upper_green = np.array([90, 255, 255])
                green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                
                total_pixels = img_cv.shape[0] * img_cv.shape[1]
                green_percentage = (cv2.countNonZero(green_mask) / total_pixels) * 100
                
                # --- NEW: THE DOUBLE-LOCK SAFETY FILTER ---
                
                # LOCK 1: Stricter Biological Filter (Bumped to 10%)
                if green_percentage < 10.0:
                    st.session_state.batch_crops.append("INVALID: Not a Plant")
                    st.session_state.batch_confidences.append(0.0)
                    st.session_state.batch_infections.append(0.0)
                    
                    disease_mask = np.zeros_like(green_mask)
                    heatmap_img = np.zeros_like(img_cv)
                    overlay = img_cv
                else:
                    # It has enough green! Now let the AI run.
                    img_resized = image_rgb.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = model.predict(img_array, verbose=0) 
                    max_confidence = np.max(predictions) * 100
                    
                    # LOCK 2: AI Confidence Gate (Must be > 65% sure it's a known leaf)
                    if max_confidence < 65.0:
                        st.session_state.batch_crops.append("INVALID: Unrecognized Object")
                        st.session_state.batch_confidences.append(max_confidence)
                        st.session_state.batch_infections.append(0.0)
                        
                        disease_mask = np.zeros_like(green_mask)
                        heatmap_img = np.zeros_like(img_cv)
                        overlay = img_cv
                    else:
                        # Passed BOTH locks! It is a real leaf and the AI knows what it is.
                        st.session_state.batch_confidences.append(max_confidence)
                        st.session_state.batch_crops.append(CLASS_NAMES[np.argmax(predictions)])
                        
                        lower_sick = np.array([0, 30, 30])
                        upper_sick = np.array([24, 255, 255])
                        disease_mask = cv2.inRange(img_hsv, lower_sick, upper_sick)
                        
                        leaf_mask = cv2.bitwise_or(green_mask, disease_mask)
                        total_leaf_pixels = cv2.countNonZero(leaf_mask)
                        if total_leaf_pixels == 0: total_leaf_pixels = 1 
                        diseased_pixels = cv2.countNonZero(disease_mask)
                        
                        infection = (diseased_pixels / total_leaf_pixels) * 100
                        st.session_state.batch_infections.append(infection)
                        
                        heatmap_img = cv2.applyColorMap(disease_mask, cv2.COLORMAP_TURBO)
                        overlay = cv2.addWeighted(img_cv, 0.7, heatmap_img, 0.5, 0)
                
                # Save visuals to memory
                st.session_state.batch_images_rgb.append(image_rgb)
                st.session_state.batch_masks.append(disease_mask)
                st.session_state.batch_overlays.append(overlay)
                
                if i == 0:
                    st.session_state.first_img_cv = img_cv
                
                my_bar.progress((i + 1) / len(images_to_process), text=f"Processed {i+1} of {len(images_to_process)} images...")
            my_bar.empty() 
            st.session_state.last_batch_id = batch_id 
            st.session_state.db_saved = False

        # --- FINANCIAL MATH ---
        num_scans = len(st.session_state.batch_infections)
        avg_infection = statistics.mean(st.session_state.batch_infections)
        dominant_crop = max(set(st.session_state.batch_crops), key=st.session_state.batch_crops.count)
        avg_confidence = statistics.mean(st.session_state.batch_confidences)
        
        if not safe_to_spray:
            ai_dosage = 0.0
        else:
            ai_dosage = 0.0 if avg_infection < 1.0 else min(avg_infection + 10.0, 100.0) 
            
        savings_percent = 100.0 - ai_dosage

        trad_total_vol = field_size * spray_vol
        trad_total_cost = trad_total_vol * chem_cost
        ai_total_vol = trad_total_vol * (ai_dosage / 100.0)
        ai_total_cost = ai_total_vol * chem_cost
        total_saved_rupees = trad_total_cost - ai_total_cost
        
       # --- SAVE TO DB AND TRIGGER ENTERPRISE NOTIFICATIONS ---
        if not st.session_state.db_saved:
            if savings_percent >= 40.0:
                # Replaced balloons with sleek, modern UI "Toasts"
                st.toast('Optimum Eco-Savings Unlocked! Drone nozzles restricted.', icon='🚁')
                st.toast(f'High ROI Detected: {savings_percent:.1f}% chemical reduction.', icon='📈')
                
            scan_type = f"Drone Batch ({num_scans} images)" if num_scans > 1 else "Single Scan"
            save_scan_to_db(dominant_crop, avg_confidence, avg_infection, ai_dosage, total_saved_rupees, scan_type)
            st.session_state.db_saved = True
        
        # --- 7. DASHBOARD DISPLAY ---
        st.markdown("---")
        
        if not safe_to_spray:
            st.error("🛑 **ENVIRONMENTAL LOCKOUT:** Live satellite data predicts high probability of rain today. All drone spraying nozzles have been administratively disabled to prevent toxic chemical runoff into local water systems.")
        
        if num_scans > 1:
            st.info(f"🚁 **Drone Sweep Complete:** Successfully analyzed {num_scans} images across the field. Displaying field averages and a representative visual sample below.")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Visual Sample")
            st.image(st.session_state.batch_images_rgb[0], use_container_width=True)
        with col2:
            st.subheader("2. Infection Mask")
            st.image(st.session_state.batch_masks[0], use_container_width=True, clamp=True, channels="GRAY")
        with col3:
            st.subheader("3. Targeting Heatmap")
            st.image(st.session_state.batch_overlays[0], use_container_width=True)

        st.markdown("---")
        col_data, col_chart = st.columns([1, 1.5])
        
        with col_data:
            st.subheader("🧠 Field Diagnostic Averages")
            
            # --- NEW UI LOGIC: Turn red if it's an invalid object! ---
            if "INVALID" in dominant_crop:
                st.error(f"**Dominant Crop:** {dominant_crop}")
            else:
                st.success(f"**Dominant Crop:** {dominant_crop}")
                
            if avg_infection < 1.0:
                st.success(f"**Average Infection:** {avg_infection:.1f}% (Healthy Field)")
            else:
                st.warning(f"**Average Infection:** {avg_infection:.1f}%")
            
            st.metric(label=f"Financial Savings (Over {field_size} Acres)", value=f"₹ {total_saved_rupees:,.2f}", delta=f"{savings_percent:.1f}% Chemical Reduction")
            
            if not safe_to_spray:
                 st.write("🔒 *Nozzles LOCKED OFF due to weather protocol.*")
            elif ai_dosage == 0:
                st.write("*Precision Nozzle is set to **OFF**. No chemicals required.*")
            else:
                st.write(f"*Applying **{ai_dosage:.1f}%** average volume across the field.*")

        with col_chart:
            st.subheader("📊 Field-Scale Spraying Action")
            fig = go.Figure(data=[
                go.Bar(name='Traditional Cost (₹)', x=['Financial Impact'], y=[trad_total_cost], marker_color='#ff4b4b'),
                go.Bar(name='AI System Cost (₹)', x=['Financial Impact'], y=[ai_total_cost], marker_color='#00cc66')
            ])
            # ANIMATED PLOTLY LAYOUT
            fig.update_layout(
                barmode='group', 
                height=300, 
                margin=dict(l=0, r=0, t=30, b=0), 
                plot_bgcolor='rgba(0,0,0,0)',
                transition=dict(duration=800, easing="cubic-in-out")
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- SECTOR-BY-SECTOR FINANCIAL BREAKDOWN ---
        if num_scans > 1:
            st.markdown("---")
            st.subheader("🔍 Sector-by-Sector Financial Breakdown")
            with st.expander(f"Click here to view detailed scans and financials for all {num_scans} field sectors"):
                
                sector_trad_cost = trad_total_cost / num_scans
                
                for idx in range(num_scans):
                    st.markdown(f"#### 📍 Sector {idx + 1}: {st.session_state.batch_crops[idx]}")
                    
                    img_inf = st.session_state.batch_infections[idx]
                    
                    if not safe_to_spray:
                        img_dosage = 0.0
                    else:
                        img_dosage = 0.0 if img_inf < 1.0 else min(img_inf + 10.0, 100.0)
                    
                    img_chem_saved = 100.0 - img_dosage
                    sector_ai_cost = sector_trad_cost * (img_dosage / 100.0)
                    sector_rupees_saved = sector_trad_cost - sector_ai_cost
                    
                    met1, met2, met3 = st.columns(3)
                    met1.metric(label="Sector Infection", value=f"{img_inf:.1f}%")
                    met2.metric(label="Required Dosage", value=f"{img_dosage:.1f}%", delta=f"{img_chem_saved:.1f}% Chemical Saved")
                    met3.metric(label="Sector Savings", value=f"₹ {sector_rupees_saved:,.2f}")
                    
                    st.write("") 
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.image(st.session_state.batch_images_rgb[idx], use_container_width=True)
                    with col_b:
                        st.image(st.session_state.batch_masks[idx], use_container_width=True, clamp=True, channels="GRAY")
                    with col_c:
                        st.image(st.session_state.batch_overlays[idx], use_container_width=True)
                    
                    st.divider() 
            
        # --- 8. REPORT GENERATION ---
        st.markdown("---")
        st.subheader("📄 Export Sweep Diagnostics")
        pdf_bytes = create_pdf_report(
            dominant_crop, 
            num_scans, 
            avg_infection, 
            ai_dosage, 
            savings_percent, 
            total_saved_rupees, 
            st.session_state.batch_images_rgb, 
            st.session_state.batch_overlays, 
            st.session_state.batch_infections,
            trad_total_cost, # NEW: Passed for the appendix ROI
            safe_to_spray    # NEW: Passed for the appendix ROI
        )
        st.download_button(label="📥 Download Drone Sweep Report (PDF)", data=pdf_bytes, file_name=f"Drone_Sweep_{dominant_crop}.pdf", mime="application/pdf", type="primary")
# ==========================================
#        PAGE 2: THE HISTORY DASHBOARD
# ==========================================
elif app_mode == "🗄️ Farm Analytics & History":
    st.title("🗄️ Field Analytics & History")
    st.markdown("A permanent cloud record of all AI field scans for long-term crop monitoring.")
    st.markdown("---")

    try:
        conn = sqlite3.connect('farm_history.db')
        df_history = pd.read_sql_query("SELECT * FROM scans ORDER BY timestamp DESC", conn)
        conn.close()
    except:
        df_history = pd.DataFrame()

    if not df_history.empty:
        total_scans = len(df_history)
        total_money_saved = df_history['saved_rupees'].sum()
        
        met1, met2 = st.columns(2)
        met1.metric(label="Total Operations Logged", value=total_scans)
        met2.metric(label="Lifetime Financial Savings", value=f"₹ {total_money_saved:,.2f}")
        
        st.markdown("---")
        
        col_hist1, col_hist2 = st.columns([2, 1])
        with col_hist1:
            st.write("**Raw Database Logs**")
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        with col_hist2:
            st.write("**Disease Severity Trend (%)**")
            st.line_chart(df_history.iloc[::-1]['severity'].reset_index(drop=True))
            
        if st.button("🗑️ Clear All Farm History"):
            conn = sqlite3.connect('farm_history.db')
            c = conn.cursor()
            c.execute("DELETE FROM scans")
            conn.commit()
            conn.close()
            st.success("Database cleared! Refresh the page.")
    else:
        st.info("No scan history yet. Go to the 'AI Field Scanner' tab and upload an image to start building your database!")