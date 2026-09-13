# Semantic Image Search Engine 
## ⚙️ Prerequisites & Installation

Open your terminal and install the following dependencies 

```powershell
# 1. Create and activate the virtual environment with your custom name
conda create --name image_search python=3.8 jupyter notebook numpy matplotlib numba scikit-learn nltk=3.6.5 -c conda-forge -y
conda activate image_search

# 2. Install libraries and frameworks
pip install mygrad mynn noggin gensim cogworks-data streamlit requests pillow tqdm

For windows only: 
# 3. Fix Windows PyTorch DLL linkage errors (Run inside terminal)
winget install -e --id Microsoft.VCRedist.2015+.x64

# 4. Enable Conda PowerShell script execution permissions
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
--------------



Follow these steps in order to download data, train the model, and launch the web interface:

### Step 1: Train the Model & Download GloVe Vectors
Run `main.py`. This script automatically downloads the required 250MB GloVe dataset via Gensim's secure servers, fits the model weights, and generates your processed image database files (`model_weights.npz` and `all_image.pkl`):

```powershell
python main.py
```
*Note: Wait a few minutes during the initial run for the automated data download to reach 100%.*

### Step 2: Launch the Web UI
Once the training script completes and finishes processing, run this command to start the interface where you enter your query. 

```
streamlit run ui.py
```

- Type your search string (e.g., *“horses on a beach”*) into the input field and press **Search** to view your ranked matching image results.
- It might take a while to load
