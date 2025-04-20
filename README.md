# RebarVista: AI Rebar Column Analysis Web Application
A tool for rebar detection and volume calculation using Detectron2 and Mask R-CNN. 
## Setup Instructions
1. Clone the Detectron2 repository:
   ```bash
   git clone https://github.com/facebookresearch/detectron2
   ```
2. Download the required model files from our Google Drive repository:
   - Access all model files here: [Web App Models](https://drive.google.com/drive/folders/1bXmBKyZlDxDyZifHVBVSFXJZoFofEVYZ?usp=sharing)
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
![Rebar Detection Example 1](Example%20Output%20Images/output1.png)
*Rebar detection with instance segmentation*
![Rebar Detection Example 2](Example%20Output%20Images/output2.png)
*Rebar detection with instance segmentation*

## Model Improvement Note
This project has significant room for improvement as the current model requires additional fine-tuning during training. Current limitations include:
- Some images with rebars are not detected at all
- In other cases, rebars are detected but with inaccurate approximations of dimensions and volume
- Performance varies significantly depending on image quality, lighting conditions, and rebar arrangement complexity
- The model was trained exclusively on 2D datasets, limiting its ability to accurately estimate volumetric properties

Future work should focus on:
1. Training with 3D datasets for more accurate detection and volume estimation
2. Exploring alternative segmentation approaches:
   - Semantic segmentation for improved pixel-level classification
   - Panoptic segmentation for better instance differentiation and boundary detection
3. Expanding the training dataset with more diverse rebar configurations
4. Implementing additional data augmentation techniques
5. Exploring hyperparameter optimization
6. Potentially testing alternative model architectures beyond Mask R-CNN

## Important Notes
- This is a development server and should not be used in production.
