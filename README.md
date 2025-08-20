# Deforestation Tracker (Python 3.11) - Jude Peach
A satellite image analysis tool used to highlight signs of possible deforestation sites in the rainforest.
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
