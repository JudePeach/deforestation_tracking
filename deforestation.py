"""
Deforestation analysis tool (WIP) crated by Jude Peach - version 1.0 as of (19-08-2025):

    - This tool allows users to draw an area of interest (AOI) on the globe, and compute forest/vegetation loss signals using NDVI.
    - NDVI is the Normalized Difference Vegetation Index, a common index for measuring vegetation density from satellite images.
    - Prerequisites and set up:
        1. Install the required packages using `pip install -r requirements.txt` or `pip install PyQt5 PyQtWebEngine earthengine-api geemap folium` (in a python 3.11 env).
        2. Authenticate with Google Earth Engine using `earthengine authenticate` (follow the instructions in your browser).
        3. (not required) If you have multiple google earth engine projects, specify the project ID in the `ee.Initialize(project='your_project_id')` (line 35).
        4. Run the script using `python deforestation.py` (ensure you have a GUI environment available).
    - Usage:
        1. Draw a rectangle on the map to select your area of interest (AOI) by pressing the square button then clicking and dragging.
        2. Select your start and end years to compare between - recommended to use 2022 onwards for best results.
        3. Click compute to start the analysis - large AOIs may take a while (5-10 minutes).
        4. The map will change highlighting your chosen AOI, the AOI will have multiple layers that you can toggle on and off:
            - Raw satellite imagery before and after the selected years.
            - NDVI visualisation before and after the selected years (dark green = high NDVI, lighter green = lower NDVI).
            - Deforestation mask (red) showing areas where NDVI dropped significantly (threshold of -0.1).
        5. The statistics area will show the average NDVI before and after, and the area of forest loss in hectares.

    - Authors recommendation - once the analysis is complete zoom into red areas, toggle off all layers except the satellite before layer, then toggle on the satellite after layer
    to see the changes in the satellite imagery. Use your own judgement to determine if the area has been deforested or not, as NDVI is not perfect and can be affected by other factors such as seasonal changes, cloud cover, etc.
    - Note: This tool is a work in progress and may not be perfect.

    - Future goals:
        1. Add more visualisations and statistics to help users understand the data better.
        2. Improve the UI and user experience.
        3. Add machine learning techniques to automatically detect deforestation patterns and trends form the ndvi differences (no need for human analysis).
        4. Make the tool automatically generate reports of potential areas of deforestation based on the analysis - i.e. scan the Amazon weekly for any new areas worth noting.
        5. Host the tool on a web server.
"""

import sys, os, tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

# PyQt handles the GUI of the application, QThread allows the application logic to run in a background thread so the UI remains responsive/alive
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QTextEdit, QHBoxLayout, QSpinBox, QLabel, QProgressBar, QStackedWidget
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QUrl

# Geospatial libraries for earth engine and map processing
import ee
import folium
from folium.plugins import Draw
import geemap.foliumap as geemap

import json, shutil, datetime

import random

# joblib to import my ML model
import joblib
import pandas as pd
import numpy as np

CLASSIFIER = joblib.load("deforestation_model.pkl")
SCALER = joblib.load("scaler.pkl")

CASES_DIR = os.path.join(os.getcwd(), "deforestation_cases")
os.makedirs(CASES_DIR, exist_ok=True)
CASES_FILE = os.path.join(CASES_DIR, "cases.json")

# ------------------------------
# Earth Engine init (ensure you've authenticated once via ee.Authenticate())
# Note: calling Initialize inside the worker thread as well for thread-safety
# ------------------------------
ee.Initialize(project='1001479155259') # PLACE YOUR PROJECT ID HERE (keep this hidden!!!)

# ------------------------------
# Helpers for EE processing
# ------------------------------
def mask_s2_clouds(image):
    """
    Masks clouds and cloud shadows from sentinel-2 satellite images. This is needed to prevent clouds etc being identified as deforestation!!

    Params: source image without masking applied
    Returns: masked image with clouds and shadows removed

    """
    scl = image.select("SCL")
    mask = (
        scl.neq(3)  # cloud shadow
        .And(scl.neq(8))  # clouds
        .And(scl.neq(9))  # high-prob clouds
        .And(scl.neq(10)) # thin cirrus
        .And(scl.neq(11)) # snow/ice
        .And(scl.neq(7))  # unclassified (often noisy)
    )
    return image.updateMask(mask)


def compute_ndvi(image):
    """
    Computes the Normalized Difference Vegetation Index from a sentinel-2 image:
        NDVI = (NIR - Red) / (NIR + Red), for sentinel-2 NIR is Band 8(B8) and Red is Band 4(B4)
    
    Params: source image with bands 8 and 4
    Returns: NDVI image with vals between -1 and 1
    """
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')


# ------------------------------
# JS bridge object exposed to the web page
# ------------------------------
class Bridge(QObject):
    """
    An object which allows JS to communicate with Python
    When the rectangle bounds are changed by the user, this emits a signal with the new AOI so that the UI can update accordingly.
    """
    aoiChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.aoi: Optional[Tuple[float, float, float, float]] = None  # (min_lat, min_lon, max_lat, max_lon)

    @pyqtSlot(float, float, float, float)
    def setAOI(self, min_lat, min_lon, max_lat, max_lon):
        # Store AOI and notify UI
        self.aoi = (min_lat, min_lon, max_lat, max_lon)
        self.aoiChanged.emit()


# ------------------------------
# Background worker for heavy EE computations (runs off the UI thread)
# ------------------------------
@dataclass
class JobParams:
    aoi: Tuple[float, float, float, float]
    start_year: int
    end_year: int


class EEWorker(QObject):
    """
    Runs EE computation in a seperate thread to stop UI being interrupted.

    Signals: finished, error, progress
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # (0-100, message)

    def __init__(self, params: JobParams):
        super().__init__()
        self.params = params

    def _emit(self, pct: int, msg: str):
        self.progress.emit(pct, msg)

    def run(self):
        """
        Main function performing the analysis and calculations.
        
        Steps:
        1. Initialize Earth Engine (if not already done).
        2. Build the AOI geometry from the provided coordinates along with time filtering being applied.
        3. Creates before and after composites from Sentinel-2 images.
        4. Computes NDVI for both composites (before and after) as well as a loss mask between them both.
        5. Computes summary statistics"
        6. Renders a map with the results and saves it to an HTML file.

        - The self.emits throughout this method updates the progress bar in the UI as each step completes/starts.
        """
        try:
            self._emit(2, "Initializing Earth Engine…")
            # Initialize in this thread as well (safer across threads)
            try:
                ee.Initialize(project='1001479155259')
            except Exception:
                # In case already initialized or project param differs, try plain init
                ee.Initialize()

            # Get the AOI and years from the params passed + some error checking/validation
            min_lat, min_lon, max_lat, max_lon = self.params.aoi
            start_year = int(self.params.start_year)
            end_year = int(self.params.end_year)
            if end_year < start_year:
                raise ValueError("End Year must be >= Start Year")

            self._emit(5, "Building AOI & filters…")
            aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], geodesic=False)

            # list of the bands we want to use so that we can filter out the ones that are not needed
            bands_to_use = ['B4', 'B3', 'B2', 'B8', 'SCL']
            start_start = f"{start_year}-03-01"
            start_end   = f"{start_year}-09-30"
            end_start   = f"{end_year}-03-01"
            end_end     = f"{end_year}-09-30"

            self._emit(15, "Creating Sentinel-2 composites…")
            base_ic = ee.ImageCollection('COPERNICUS/S2_SR')

            # Filter the image collection by date and bounds, apply cloud masking, and select bands
            # We use median to create a composite image for the before and after periods from the image collections
            before = (base_ic
                      .filterBounds(aoi)
                      .filterDate(start_start, start_end)
                      .map(mask_s2_clouds) #applying cloud mask function from earlier
                      .select(bands_to_use)
                      .median() #takes the per pixel median of the image collection
                      .clip(aoi))

            after = (base_ic
                     .filterBounds(aoi)
                     .filterDate(end_start, end_end)
                     .map(mask_s2_clouds) #applying cloud mask function from earlier
                     .select(bands_to_use)
                     .median() #takes the per pixel median of the image collection
                     .clip(aoi))

            # Compute NDVI for each composite as well as creating the deforestation mask (what will be later shown in red)
            self._emit(35, "Computing NDVI & change…")
            before_ndvi = compute_ndvi(before)
            after_ndvi = compute_ndvi(after)
            ndvi_diff = after_ndvi.subtract(before_ndvi).rename('NDVI_DIFF')
            deforestation_mask = ndvi_diff.lt(-0.1)  # threshold for loss

            # Stat calculations
            self._emit(55, "Computing summary statistics…")
            reducer = ee.Reducer.mean()
            before_mean = before_ndvi.reduceRegion(
                reducer=reducer, geometry=aoi, scale=30, maxPixels=1e9
            ).get('NDVI')
            after_mean = after_ndvi.reduceRegion(
                reducer=reducer, geometry=aoi, scale=30, maxPixels=1e9
            ).get('NDVI')

            # Area of loss in hectares
            loss_image = deforestation_mask.rename('loss').selfMask().multiply(ee.Image.pixelArea())
            loss_area_m2 = loss_image.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
            ).get('loss')

            # Pull to client (still blocking but on worker thread)
            before_mean = ee.Number(before_mean).getInfo()
            after_mean = ee.Number(after_mean).getInfo()
            loss_area_m2 = (ee.Number(loss_area_m2).getInfo() if loss_area_m2 is not None else 0)
            loss_area_ha = (loss_area_m2 or 0) / 10000.0

            self._emit(75, "Rendering map…")
            # Visualization params (keep some layers hidden by default for speed)
            true_color_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3500}
            ndvi_vis = {'min': 0, 'max': 1, 'palette': ['white', 'lightgreen', 'green', 'darkgreen']}

            # Use a slightly higher zoom for large AOIs - saves time - could be changed to a user setting
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Prefer Canvas can speed up vector rendering
            fmap = geemap.Map(center=[center_lat, center_lon], zoom=8, basemap="SATELLITE")

            # Add lightweight layers first; show only After true color by default
            fmap.addLayer(after, true_color_vis, 'Satellite After', shown=True)
            fmap.addLayer(before, true_color_vis, 'Satellite Before', shown=False)
            fmap.addLayer(before_ndvi, ndvi_vis, 'NDVI Before', shown=False)
            fmap.addLayer(after_ndvi, ndvi_vis, 'NDVI After', shown=False)
            fmap.addLayer(
                deforestation_mask.updateMask(deforestation_mask),
                {'palette': ['red'], 'opacity': 0.5},
                'Deforestation (NDVI drop > 0.1)',
                shown=True
            )
            fmap.addLayer(ee.Image().paint(aoi, 1, 3), {'palette': ['blue']}, 'AOI Outline', shown=True)
            fmap.addLayerControl()

            # Add a simple NDVI legend strip (lightweight HTML)
            legend_html = """
            <div style='position: fixed; bottom: 50px; left: 50px; width: 220px; height: 30px;
                        background: linear-gradient(to right, white, lightgreen, green, darkgreen);
                        border: 2px solid grey; z-index:9999; font-size:14px;'>
              <div style='display:flex; justify-content:space-between; padding:0 5px;'>
                <span>0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.8</span><span>1</span>
              </div>
            </div>
            """
            fmap.get_root().html.add_child(folium.Element(legend_html))

            out_html = os.path.join(tempfile.gettempdir(), "deforestation_map.html")
            fmap.save(out_html)

            self._emit(95, "Finalizing…")
            results = {
                "before_mean": float(before_mean) if before_mean is not None else None,
                "after_mean": float(after_mean) if after_mean is not None else None,
                "loss_area_ha": float(loss_area_ha),
                "aoi": (min_lat, min_lon, max_lat, max_lon),
                "start_year": start_year,
                "end_year": end_year,
                "out_html": out_html,
            }
            self._emit(100, "Done")
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------
# Main App (includes the UI and event handlers)
# ------------------------------
class DeforestationApp(QWidget):
    """
    GUI app for the tool + wires up the JS bridge to allow the user to draw an AOI on the map.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌍 Deforestation Analysis (Draw AOI)")
        self.setGeometry(100, 100, 1200, 800)

        # Connect to the java script bridge to allow the map to update in the UI
        self.bridge = Bridge()
        self.bridge.aoiChanged.connect(self.on_aoi_changed)

        # UI Layout - using stacks for handling seperate pages/views
        self.stack = QStackedWidget()
        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.stack)
        
        self.main_page = QWidget()
        root = QVBoxLayout(self.main_page)

         # Controls row: years + compute
        controls = QHBoxLayout()
        root.addLayout(controls)
        
        controls.addWidget(QLabel("Start Year:"))
        self.start_year = QSpinBox()
        self.start_year.setRange(2016, 2035)
        self.start_year.setValue(2020)
        controls.addWidget(self.start_year)

        controls.addWidget(QLabel("End Year:"))
        self.end_year = QSpinBox()
        self.end_year.setRange(2016, 2035)
        self.end_year.setValue(2025)
        controls.addWidget(self.end_year)

        self.btn_compute = QPushButton("Compute Deforestation")
        self.btn_save_case = QPushButton("Save this case")
        self.btn_save_case.setEnabled(False)  # Initially disabled until computation is done
        self.btn_save_case.clicked.connect(self.save_case)
        self.btn_compute.clicked.connect(self.compute_deforestation)
        controls.addWidget(self.btn_compute)
        controls.addWidget(self.btn_save_case)

        self.btn_cases = QPushButton("Open Cases")
        self.btn_cases.clicked.connect(self.open_cases)
        controls.addWidget(self.btn_cases)


        # Progress bar (determinate with text) - starts off hidden and becomes visible when computation starts
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Stats area
        self.stats = QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setPlaceholderText("Draw a rectangle on the map, then click Compute.")
        root.addWidget(self.stats)

        # Web view (map)
        self.map_view = QWebEngineView()
        root.addWidget(self.map_view)

        # QWebChannel hookup
        self.channel = QWebChannel()
        self.channel.registerObject('bridge', self.bridge)
        self.map_view.page().setWebChannel(self.channel)

        # Load the initial map that allows drawing an AOI
        self.load_selection_map()

        self.stack.addWidget(self.main_page)

        # Threading members
        self.thread: Optional[QThread] = None
        self.worker: Optional[EEWorker] = None

    # --------------------------
    # Map with drawing tools (folium based + embedded JS)
    # --------------------------
    def load_selection_map(self):
        # Base map with drawing options
        fmap = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")
        Draw(
            draw_options={
                "polyline": False, "polygon": False, "circle": False,
                "marker": False, "circlemarker": False, "rectangle": True
            },
            edit_options={"edit": True, "remove": True}
        ).add_to(fmap)

        # Inject QWebChannel + JS to pass rectangle bounds to Python
        map_var = fmap.get_name()  # e.g., 'map_12345abcd'
        js = f"""
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        // Wait for Qt WebChannel to be ready
        new QWebChannel(qt.webChannelTransport, function(channel) {{
            const bridge = channel.objects.bridge;
            const map = {map_var};

            const drawnItems = new L.FeatureGroup();
            map.addLayer(drawnItems);

            map.on(L.Draw.Event.CREATED, function (e) {{
                drawnItems.clearLayers();
                const layer = e.layer;
                drawnItems.addLayer(layer);

                if (layer.getBounds) {{
                    const b = layer.getBounds();
                    // Send bounds to Python bridge
                    bridge.setAOI(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
                }}
            }});

            // If user edits the rectangle, update AOI again
            map.on('draw:edited', function(e) {{
                e.layers.eachLayer(function (layer) {{
                    if (layer.getBounds) {{
                        const b = layer.getBounds();
                        bridge.setAOI(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
                    }}
                }});
            }});
        }});
        </script>
        """
        fmap.get_root().html.add_child(folium.Element(js))

        tmp_path = os.path.join(tempfile.gettempdir(), "select_aoi_map.html")
        fmap.save(tmp_path)
        self.map_view.load(QUrl.fromLocalFile(tmp_path))

    # --------------------------
    # UI event handlers
    # --------------------------
    def on_aoi_changed(self):
        min_lat, min_lon, max_lat, max_lon = self.bridge.aoi
        self.stats.setText(
            f"AOI selected:\n"
            f"  South (min_lat): {min_lat:.6f}\n"
            f"  West  (min_lon): {min_lon:.6f}\n"
            f"  North (max_lat): {max_lat:.6f}\n"
            f"  East  (max_lon): {max_lon:.6f}\n\n"
            f"Set your years and click Compute."
        )

    def compute_deforestation(self):
        """ Starts the deforestation analysis computation with background processing via QThread and EEworker...."""
        if not self.bridge.aoi:
            self.stats.setText("⚠️ Please draw a rectangle first.")
            return

        params = JobParams(
            aoi=self.bridge.aoi,
            start_year=int(self.start_year.value()),
            end_year=int(self.end_year.value()),
        )

        # Disable button & show progress
        self.btn_compute.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setFormat("Starting computation...")

        # Spin up worker thread to handle the google earth enginer computations (these take longer)
        self.thread = QThread()
        self.worker = EEWorker(params)
        self.worker.moveToThread(self.thread)

        # Wire signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_computation_done)
        self.worker.error.connect(self.on_computation_error)
        self.worker.progress.connect(self.on_progress)
        # Ensure thread stops
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def open_cases(self):
        """ Opens a new tab with a list of the open cases of possible deforestation
            The user can click on each case to view the saved map and statistics linked to the aoi.
            They can analyse cases and use machine learning to detect if they show deforestation.
            From there they can report/raise the case.
            """
        # Create a new QWidget for the cases tab
        self.cases_page = QWidget()
        layout = QVBoxLayout(self.cases_page)

        # Add a back button
        back_btn = QPushButton("← Back")
        layout.addWidget(back_btn)

        # Add a placeholder for cases list
        cases_label = QLabel("Open Cases:")
        layout.addWidget(cases_label)

        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, "r") as f:
                cases = json.load(f)
        else:
            cases = []
        
        if len(cases) == 0:
            cases_label.setText("No cases saved yet. Compute a deforestation case first.")
            return
        else:
            for case in cases:
                btn = QPushButton(
                f"{case['id']} | {case['start_year']}–{case['end_year']} | Loss: {case['loss_area_ha']:.2f} ha"
                )
                layout.addWidget(btn)

                def open_case(*args, case=case):
                    self.stats.setText(
                        f"AOI: {case['aoi']}\n"
                        f"Time Period: {case['start_year']} – {case['end_year']}\n"
                        f"NDVI Before: {case['before_mean']:.3f}\n"
                        f"NDVI After: {case['after_mean']:.3f}\n"
                        f"Forest Loss: {case['loss_area_ha']:.2f} ha"
                    )
                    self.map_view.load(QUrl.fromLocalFile(case["out_html"]))
                    self.stack.setCurrentWidget(self.main_page)

                btn.clicked.connect(open_case)

        if self.stack.indexOf(self.cases_page) == -1:
            self.stack.addWidget(self.cases_page)
        self.stack.setCurrentWidget(self.cases_page)

        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_page))

    def save_case(self):
        """ Saves the current case with the AOI, years, and results to a file.
            This is a placeholder function that can be expanded to save to a database or file system.
            """
        if not hasattr(self, "last_results") or not self.last_results:
            self.stats.setText("⚠️ No results to save. Please compute first.")
            return
        
        results = self.last_results

        # Copy the map HTML to permanent storage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = f"case_{timestamp}"
        map_path = os.path.join(CASES_DIR, f"{case_id}.html")
        shutil.copy(results["out_html"], map_path)
        results["out_html"] = map_path
        results["id"] = case_id

        # Load existing cases
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, "r") as f:
                cases = json.load(f)
        else:
            cases = []

        cases.append(results)
        
        # Save updated cases
        with open(CASES_FILE, "w") as f:
            json.dump(cases, f, indent=2)

        self.stats.setText(f"✅ Case saved: {case_id}")

    
    # --------------------------
    # Worker callbacks
    # --------------------------
    def on_progress(self, pct: int, message: str):
        self.progress.setValue(pct)
        self.progress.setFormat(f"{message} (%p%)")
        QApplication.processEvents()

    def on_computation_done(self, results: dict):

        #Persist the results (so they can be saved if needed)
        self.last_results = results

        self.btn_compute.setEnabled(True)
        self.btn_save_case.setEnabled(True)
        self.progress.setVisible(False)

        def fmt(x, nd=3):
            try:
                return f"{x:.{nd}f}"
            except Exception:
                return "N/A"

        min_lat, min_lon, max_lat, max_lon = results["aoi"]
        
        # Prep feats for the ML model:
        aoi_area_ha = (abs(max_lat - min_lat) * abs(max_lon - min_lon)) * 111 * 111
        diff_mean = (results["after_mean"] or 0) - (results["before_mean"] or 0)

        aoi_dict = {
                "before_mean": results["before_mean"],
                "after_mean": results["after_mean"],
                "loss_area_ha": results["loss_area_ha"],
                "aoi_area_ha": aoi_area_ha,
                "diff_mean": diff_mean
                }

        try:
            prediction = predict_deforestation(aoi_dict)
            pred_text = "🌲 Likely Deforestation" if prediction == 1 else "✅ No Significant Deforestation"
        except Exception as e:
            pred_text = f"PREDICTION ERROR: {e}"

        self.stats.setText(
            f"AOI: ({min_lat:.6f}, {min_lon:.6f}) → ({max_lat:.6f}, {max_lon:.6f})\n"
            f"Time Period: {results['start_year']} – {results['end_year']}\n"
            f"NDVI Before: {fmt(results['before_mean'])}\n"
            f"NDVI After:  {fmt(results['after_mean'])}\n"
            f"Forest Loss: {results['loss_area_ha']:.2f} ha\n\n"
            f"ML Prediction: {pred_text}\n\n"
            "NDVI guide:\n"
            "  0.8–1.0: very dense vegetation\n"
            "  0.6–0.8: healthy vegetation\n"
            "  0.4–0.6: sparse vegetation\n"
            "  0.2–0.4: very sparse / shrubs\n"
            "  0–0.2: bare soil / stressed\n"
            "  <0: water/snow/clouds"
        )

        self.map_view.load(QUrl.fromLocalFile(results["out_html"]))

    def on_computation_error(self, msg: str):
        self.btn_compute.setEnabled(True)
        self.progress.setVisible(False)
        self.stats.setText(f"❌ Error: {msg}")

def predict_deforestation(aoi_dict):
    """
        Takes in an aoi dict with stats about the aoi and uses ML to predict deforestation.
    """
    df = pd.DataFrame([aoi_dict])

    # Add our engineered features the model needs
    df["diff_weighted"] = df["diff_mean"] * np.log1p(df["aoi_area_ha"])
    df["diff_area_interaction"] = df["diff_mean"] * df["aoi_area_ha"]

    # make sure cols are in same order as in our training
    df = df.reindex(columns=X.columns, fill_value=0)

    # Scale and predict
    df_scaled = SCALER.transform(df)
    prediction = CLASSIFIER.predict(df_scaled)

    return prediction[0]

def run_aoi_ml(min_lat, min_lon, max_lat, max_lon):

    print("Running deforestation analysis for AOI:", min_lat, min_lon, max_lat, max_lon)
    try:
            
            # Initialize in this thread as well (safer across threads)
            try:
                ee.Initialize(project='1001479155259')
            except Exception:
                # In case already initialized or project param differs, try plain init
                ee.Initialize()

            # Get the AOI and years from the params passed + some error checking/validation
            
            start_year = 2022
            end_year = 2024
            if end_year < start_year:
                raise ValueError("End Year must be >= Start Year")

            
            aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], geodesic=False)

            # list of the bands we want to use so that we can filter out the ones that are not needed
            bands_to_use = ['B4', 'B3', 'B2', 'B8', 'SCL']
            start_start = f"{start_year}-03-01"
            start_end   = f"{start_year}-09-30"
            end_start   = f"{end_year}-03-01"
            end_end     = f"{end_year}-09-30"

            
            base_ic = ee.ImageCollection('COPERNICUS/S2_SR')

            # Filter the image collection by date and bounds, apply cloud masking, and select bands
            # We use median to create a composite image for the before and after periods from the image collections
            before = (base_ic
                      .filterBounds(aoi)
                      .filterDate(start_start, start_end)
                      .map(mask_s2_clouds) #applying cloud mask function from earlier
                      .select(bands_to_use)
                      .median() #takes the per pixel median of the image collection
                      .clip(aoi))

            after = (base_ic
                     .filterBounds(aoi)
                     .filterDate(end_start, end_end)
                     .map(mask_s2_clouds) #applying cloud mask function from earlier
                     .select(bands_to_use)
                     .median() #takes the per pixel median of the image collection
                     .clip(aoi))

            # Compute NDVI for each composite as well as creating the deforestation mask (what will be later shown in red)
            
            before_ndvi = compute_ndvi(before)
            after_ndvi = compute_ndvi(after)
            ndvi_diff = after_ndvi.subtract(before_ndvi).rename('NDVI_DIFF')
            deforestation_mask = ndvi_diff.lt(-0.1)  # threshold for loss

            # Stat calculations
            
            reducer = ee.Reducer.mean()
            before_mean = before_ndvi.reduceRegion(
                reducer=reducer, geometry=aoi, scale=30, maxPixels=1e9
            ).get('NDVI')
            after_mean = after_ndvi.reduceRegion(
                reducer=reducer, geometry=aoi, scale=30, maxPixels=1e9
            ).get('NDVI')

            # Area of loss in hectares
            loss_image = deforestation_mask.rename('loss').selfMask().multiply(ee.Image.pixelArea())
            loss_area_m2 = loss_image.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
            ).get('loss')

            # Pull to client (still blocking but on worker thread)
            before_mean = ee.Number(before_mean).getInfo()
            after_mean = ee.Number(after_mean).getInfo()
            loss_area_m2 = (ee.Number(loss_area_m2).getInfo() if loss_area_m2 is not None else 0)
            loss_area_ha = (loss_area_m2 or 0) / 10000.0

            
            # Visualization params (keep some layers hidden by default for speed)
            true_color_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3500}
            ndvi_vis = {'min': 0, 'max': 1, 'palette': ['white', 'lightgreen', 'green', 'darkgreen']}

            # Use a slightly higher zoom for large AOIs - saves time - could be changed to a user setting
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Prefer Canvas can speed up vector rendering
            fmap = geemap.Map(center=[center_lat, center_lon], zoom=8, basemap="SATELLITE")

            # Add lightweight layers first; show only After true color by default
            fmap.addLayer(after, true_color_vis, 'Satellite After', shown=True)
            fmap.addLayer(before, true_color_vis, 'Satellite Before', shown=False)
            fmap.addLayer(before_ndvi, ndvi_vis, 'NDVI Before', shown=False)
            fmap.addLayer(after_ndvi, ndvi_vis, 'NDVI After', shown=False)
            fmap.addLayer(
                deforestation_mask.updateMask(deforestation_mask),
                {'palette': ['red'], 'opacity': 0.5},
                'Deforestation (NDVI drop > 0.1)',
                shown=True
            )
            fmap.addLayer(ee.Image().paint(aoi, 1, 3), {'palette': ['blue']}, 'AOI Outline', shown=True)
            fmap.addLayerControl()

            # Add a simple NDVI legend strip (lightweight HTML)
            legend_html = """
            <div style='position: fixed; bottom: 50px; left: 50px; width: 220px; height: 30px;
                        background: linear-gradient(to right, white, lightgreen, green, darkgreen);
                        border: 2px solid grey; z-index:9999; font-size:14px;'>
              <div style='display:flex; justify-content:space-between; padding:0 5px;'>
                <span>0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.8</span><span>1</span>
              </div>
            </div>
            """
            fmap.get_root().html.add_child(folium.Element(legend_html))

            out_html = os.path.join(tempfile.gettempdir(), "deforestation_map.html")
            fmap.save(out_html)

            
            cloud_pct = estimate_cloud_coverage(after, aoi)

            results = {
                "before_mean": float(before_mean) if before_mean is not None else None,
                "after_mean": float(after_mean) if after_mean is not None else None,
                "loss_area_ha": float(loss_area_ha),
                "cloud_pct": float(cloud_pct),
                "aoi": (min_lat, min_lon, max_lat, max_lon),
                "start_year": start_year,
                "end_year": end_year,
                "out_html": out_html,
                        }
            
    except Exception as e:
        print(e)

    return results

def random_aoi(size_deg=0.1):
    """ 
        Computes a random area of interest for processing for our training data.
    """
    # Lat/lon ranges roughly in high deforestation areas
    lat_range = (-9, -3)      # hotspots in Amazonas, Mato Grosso
    lon_range = (-65, -55)    # western Brazil
    min_lat = random.uniform(*lat_range)
    min_lon = random.uniform(*lon_range)
    max_lat = min_lat + size_deg
    max_lon = min_lon + size_deg
    return (min_lat, min_lon, max_lat, max_lon)

def save_to_ml_dataset(results):
        """ Saves the current case data to a CSV file for training a machine learning model.
            """
        print("Saving case data to train ML model (csv)...")

        
        #label = self.label_combo.currentText()

        #Make a unique ID for each sample
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = f"sample_{timestamp}"

        label = assign_label(results)

        new_row = [
            sample_id, results['start_year'], results['end_year'],
            results['aoi'][0], results['aoi'][2], results['aoi'][1], results['aoi'][3],
            results.get('aoi_area_ha', 0.0), results.get('loss_area_ha', 0.0), results.get('loss_pct', 0.0),
            results.get('before_mean', None), results.get('after_mean', None), results.get('after_mean', 0) - results.get('before_mean', 0),
            label
        ]

        # Open the csv training file in append mode
        train_file = open("deforestation_cases/training_data.csv", "a")

        try:
            # Write the new row
            train_file.write(",".join(map(str, new_row)) + "\n")
            
        except Exception as e:
            print(e)
        
        print("Case written to training dataset!: {sample_id}")
    
def automate_train_data():
    num_samples = 2300
    while num_samples > 0:
        aoi = random_aoi()
        results = run_aoi_ml(*aoi)
        if results:
            save_to_ml_dataset(results)
            num_samples -= 1
            print(f"Samples remaining: {num_samples}")
        else:
            print("Skipping sample due to error.")

def assign_label(results, ndvi_drop_thresh=-0.02, loss_area_thresh_ha=5, cloud_pct_thresh=20):
    """
    Assigns a label based on rules:
      1 = likely deforestation
      0 = likely no deforestation (or cloud/seasonal changes)
    """
    diff_mean = results.get("after_mean", 0) - results.get("before_mean", 0)
    loss_area = results.get("loss_area_ha", 0)
    cloud_pct = results.get("cloud_pct", 0)  # you need to compute this during run_aoi_ml

    print("Diff Mean:", diff_mean, "Loss Area (ha):", loss_area, "Cloud %:", cloud_pct)
    print("Thresholds - NDVI drop:", ndvi_drop_thresh, "Loss Area (ha):", loss_area_thresh_ha, "Cloud %:", cloud_pct_thresh)
    if diff_mean < ndvi_drop_thresh and loss_area > loss_area_thresh_ha and cloud_pct < cloud_pct_thresh:
        return 1  # deforestation
    else:
        return 0  # not deforestation / uncertain


def estimate_cloud_coverage(image, aoi):
    """
    Returns the percent of AOI covered by clouds.
    image: Sentinel-2 image with 'SCL' band
    """
    scl = image.select('SCL')
    cloud_pixels = scl.eq(8).Or(scl.eq(9))
    cloud_area = cloud_pixels.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).get('SCL')

    total_area = ee.Image.pixelArea().reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).get('area')

    cloud_pct = ee.Number(cloud_area).divide(total_area).multiply(100)
    return cloud_pct.getInfo() if cloud_pct else 0

# ------------------------------
# Entrypoint - starts QT app and shows user main window upon code running.
# ------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        # Run automation mode
        automate_train_data()
    else:
        # Default: start the Qt app
        app = QApplication(sys.argv)
        w = DeforestationApp()
        w.show()
        sys.exit(app.exec_())
