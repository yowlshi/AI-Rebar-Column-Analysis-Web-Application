SETTING UP:
1. Download Detectron2 repo 'https://github.com/facebookresearch/detectron2'
2. Download R-101.pkl and model_final.pth here https://drive.google.com/drive/folders/1bXmBKyZlDxDyZifHVBVSFXJZoFofEVYZ?usp=sharing 
3. change the '_BASE_' model path in 'models/mask_rcnn_R_101_FPN_3x.yaml' with the path where you downloaded the Base-RCNNN-FPN.yaml model.
4. also change for the weights path in 'models/mask_rcnn_R_101_FPN_3x.yaml' with the path where you downloaded the R-101.pkl model.
5. create a virtual environment python inside the folder of the web app via vscode terminal.


HOW TO USE:
1. Just activate the virtual environment python then run the 'app.py' then just copy the link generated
2. It should display like this:
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
 * Running on http://1**.*.*.*:****
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
 * Debugger PIN: ***-***-***

