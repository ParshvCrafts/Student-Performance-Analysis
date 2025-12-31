# Deployment Guide - Student Performance Predictor

## Problem Solved

This deployment configuration fixes the `'SimpleImputer' object has no attribute '_fill_dtype'` error caused by scikit-learn version incompatibility between training and deployment environments.

## Solutions Implemented

### 1. **Pinned Dependencies** (`requirements.txt`)
- All package versions are now explicitly pinned
- Ensures consistent environment across local and production
- Prevents automatic upgrades that break compatibility

### 2. **Automatic Model Retraining** (`build.sh`)
- Models are automatically retrained during deployment
- Ensures pickle files match the deployment environment's sklearn version
- Runs only if artifacts are missing

### 3. **Production Server** (`start.sh`)
- Uses Gunicorn instead of Flask's development server
- Better performance and reliability
- Proper timeout handling (120 seconds)

### 4. **Enhanced Error Handling** (`src/utils.py`)
- Detailed logging for version mismatch issues
- Clear error messages for troubleshooting

## Render Deployment Instructions

### Option 1: Quick Fix (Delete and Rebuild)

1. **Delete old pickle files from your repository:**
   ```bash
   git rm artifacts/model.pkl artifacts/preprocessor.pkl
   git commit -m "Remove old pickle files - will regenerate on deployment"
   git push
   ```

2. **Configure Render:**
   - Build Command: `bash build.sh`
   - Start Command: `bash start.sh`

3. **Deploy** - The build script will automatically retrain models with the correct sklearn version

### Option 2: Keep Artifacts (Force Retrain)

If you want to keep artifacts in git but force retrain:

1. **Modify build.sh** to always retrain:
   ```bash
   # Remove the condition check, always retrain
   echo "Retraining models..."
   python retrain_models.py
   ```

2. **Configure Render:**
   - Build Command: `bash build.sh`
   - Start Command: `bash start.sh`

3. **Deploy**

### Environment Variables (Optional)

Set these in Render dashboard if needed:
- `PYTHON_VERSION`: `3.10.x` (or your preferred version)
- `PORT`: Auto-assigned by Render

## Local Testing

### 1. Clean Environment Setup
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain Models
```bash
# This regenerates all pickle files with current sklearn version
python retrain_models.py
```

### 3. Run Application
```bash
# Development mode
python app.py

# Production mode (local testing)
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app:app
```

### 4. Test the Endpoint
Visit: `http://localhost:8000` (or 5000 for development mode)

## Troubleshooting

### Error: Models still incompatible
**Solution:** Delete artifacts folder on Render and redeploy

### Error: Build timeout
**Solution:** Increase build timeout in Render settings (Settings > Build & Deploy)

### Error: Module not found
**Solution:** Verify `requirements.txt` includes all dependencies and `setup.py` is configured correctly

### Error: Permission denied on .sh files
**Solution:** Make scripts executable:
```bash
chmod +x build.sh start.sh
git add build.sh start.sh
git commit -m "Make scripts executable"
git push
```

## Files Modified/Created

- ✅ `requirements.txt` - Pinned all dependencies
- ✅ `build.sh` - Automated build script
- ✅ `start.sh` - Production start script
- ✅ `retrain_models.py` - Model retraining script
- ✅ `src/utils.py` - Enhanced error handling
- ✅ `.gitignore` - Documentation for artifacts

## Architecture

```
Render Deployment Flow:
1. git push → Trigger deployment
2. build.sh → Install dependencies
3. build.sh → Check for artifacts
4. build.sh → Retrain if needed (runs retrain_models.py)
5. start.sh → Launch gunicorn server
6. App ready! ✅
```

## Best Practices Going Forward

1. **Always pin dependencies** - Never use unpinned versions in production
2. **Test locally first** - Use the retrain script before deploying
3. **Monitor logs** - Check Render logs for any warnings
4. **Version control** - Keep track of sklearn version used for training
5. **Regular updates** - Update dependencies together and retrain all models

## Quick Commands

```bash
# Retrain locally
python retrain_models.py

# Test locally with gunicorn
gunicorn --bind 0.0.0.0:8000 app:app

# Deploy to Render (after push)
git add .
git commit -m "Update deployment configuration"
git push origin main
```

## Support

If issues persist:
1. Check Render build logs
2. Verify Python version matches local environment
3. Ensure all dependencies installed successfully
4. Try manual retraining on Render console
