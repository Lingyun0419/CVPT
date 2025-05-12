## Environment
1. Create the conda environment
   ```bash
   conda create --name vit-grad-cam python=3.10
   ```

2. Activate the environment
   ```bash
   conda activate vit-grad-cam
   ```

3. Install the PyTorch
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install additional dependency
   ```bash
   pip install -r requirements.txt   
   ```
