# 🛠️ Installation Guide

## Quick Setup (Recommended)

### Step 1: Create Virtual Environment
```bash
# Make sure you're in the vector database project directory
cd /home/andy/quicksight/vectorDb

# Create virtual environment (already done for you!)
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

### Step 2: Install Packages by Quest Level

#### For Quest 1 (Basic Vector Operations) ✅ READY!
```bash
# Already installed!
pip install numpy pandas matplotlib
```

#### For Quest 2 (ChromaDB & Real Embeddings)
```bash
pip install chromadb sentence-transformers scikit-learn
```

#### For Quest 3+ (Advanced Features)
```bash
pip install faiss-cpu torch transformers openai
```

### Step 3: Test Your Installation
```bash
# Check what's ready
python check_env.py

# Test basic functionality
python quick_start.py
```

## Alternative: Install Everything at Once

```bash
# Activate environment
source venv/bin/activate

# Install all packages (may take 10-15 minutes)
pip install -r requirements.txt
```

## Environment Management

### Activate Environment (each time you work)
```bash
source venv/bin/activate
```

### Deactivate Environment (when done)
```bash
deactivate
```

### Check Status
```bash
python check_env.py
```

## Troubleshooting

### If installation fails:
1. Make sure virtual environment is activated: `source venv/bin/activate`
2. Upgrade pip: `pip install --upgrade pip`
3. Install packages one by one
4. For CUDA support: replace `faiss-cpu` with `faiss-gpu`

### System Requirements:
- Python 3.8+
- 4GB+ RAM for basic exercises
- 8GB+ RAM for advanced exercises with large models

### Package Download Sizes:
- Core packages (Quest 1): ~50MB
- Vector DB packages (Quest 2): ~500MB  
- Advanced packages (Quest 3+): ~2GB (includes PyTorch)

## Ready to Start?

✅ **You're ready for Quest 1!** Open `roadMap.md` and begin your hero's journey!

🔄 **For Quest 2+**: Run the installation commands above, then continue your adventure!