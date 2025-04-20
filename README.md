# RebarVista: AI Rebar Analysis Web Application

A tool for rebar detection and volume calculation using Detectron2 and Mask R-CNN. 

## Setup Instructions

1. Clone the Detectron2 repository:
   ```bash
   git clone https://github.com/facebookresearch/detectron2
   ```

2. Download the required model files from our Google Drive repository:
   - Access all model files here: [Web APP Models](https://drive.google.com/drive/folders/1bXmBKyZlDxDyZifHVBVSFXJZoFofEVYZ?usp=sharing)
   - You'll need both `R-101.pkl` and `model_final.pth`

3. Update the model configuration paths in `models/mask_rcnn_R_101_FPN_3x.yaml`:
   - Set the `_BASE_` path to the location of your downloaded `Base-RCNN-FPN.yaml` file
   - Set the weights path to the location of your downloaded `R-101.pkl` file

4. Create a virtual environment inside the web app folder:
   ```bash
   # Navigate to your project directory
   cd path/to/your/project

   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment if not already activated:
   ```bash
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open the link displayed in the terminal in your web browser.

## Example Terminal Output

```
(venv) PS C:\Users...> python app.py 
----- RebarVista DEBUG INFO -----
Current Working Directory: C:\Users...
Model Config Path: C:/Users/.../model/mask_rcnn_R_101_FPN_3x.yaml
Model Weights Path: C:/Users/.../model/model_final.pth
Config Exists: True
Weights Exist: True
----------------------------------
Configuration file loaded successfully.
Predictor initialized successfully.
* Serving Flask app 'app'
* Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
* Restarting with stat
----- RebarVista DEBUG INFO -----
Current Working Directory: C:\Users...
Model Config Path: C:/Users/.../model/mask_rcnn_R_101_FPN_3x.yaml 
Model Weights Path: C:/Users/.../model/model_final.pth
Config Exists: True
Weights Exist: True
----------------------------------
Configuration file loaded successfully.
Predictor initialized successfully.
* Debugger is active!
* Debugger PIN: 123-456-789
```

## Example Output Images

Here are some examples of the rebar detection results:

![Rebar Detection Example 1](AI-Rebar-Analysis-Web-Application/Example%20Output%20Images/output1.png)

![Rebar Detection Example 2](AI-Rebar-Analysis-Web-Application/Example%20Output%20Images/output2.png)

## Important Notes

- This is a development server and should not be used in production.
