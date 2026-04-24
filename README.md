# Precision-Agronomy-AI
# 🌱 AI-Powered Precision Agronomy System 🚁

An edge-to-cloud precision agriculture platform designed for autonomous UAVs. This system fuses deep learning diagnostics with OpenCV spatial masking to calculate localized crop infection severity and drive variable-rate chemical spraying.

### 🚀 Tech Stack
* **Deep Learning:** TensorFlow / Keras (MobileNetV2)
* **Computer Vision:** OpenCV (HSV Spatial Masking)
* **Frontend/Deployment:** Streamlit
* **IoT & Weather:** HTML5 Geolocation, Open-Meteo API
* **Predictive Modeling:** XGBoost, Random Forest

---

### 🧠 Core Features
1. **CNN Disease Classifier:** Utilizes a lightweight MobileNetV2 architecture trained on PlantVillage and UAV imagery for rapid edge-node inference (28 FPS).
2. **OpenCV Spatial Masking:** Dynamically isolates necrotic tissue from healthy chlorophyll to calculate the exact Infection Severity Percentage ($S_{inf}$).
3. **Double-Lock Biological Safety Filter:** A heuristic pre-filter that prevents false-positive actuations by ensuring targets meet minimum chlorophyll density and AI confidence thresholds (eliminates 98.4% of non-plant misfires).
4. **IoT Environmental Lockout:** Securely fetches real-time drone coordinates to query satellite weather data, automatically locking spray nozzles if precipitation probability exceeds 50% to prevent ecological runoff.
5. **Variable-Rate Economics:** Calculates optimal chemical dosages per field sector, yielding up to an 80% reduction in agrochemical volume compared to traditional blanket spraying.

---

### 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/subhanikhan45/Precision-Agronomy-AI.git
