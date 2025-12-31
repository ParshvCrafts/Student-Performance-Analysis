# Quick Fix Guide - Deploy to Render

## The Problem
`'SimpleImputer' object has no attribute '_fill_dtype'` - Scikit-learn version mismatch

## The Solution (3 Steps)

### Step 1: Push Updated Code
```bash
git add .
git commit -m "Fix sklearn compatibility - add auto-retrain"
git push origin main
```

### Step 2: Configure Render
In your Render service dashboard:

**Build Command:**
```
bash build.sh
```

**Start Command:**
```
bash start.sh
```

### Step 3: Deploy
Click "Manual Deploy" → "Clear build cache & deploy"

## What Happens
1. ✅ Installs pinned dependencies (sklearn 1.3-1.7)
2. ✅ Automatically retrains models with current sklearn version
3. ✅ Starts production server with gunicorn
4. ✅ Your app works! 🎉

## Verify It's Working
Visit: `https://your-app.onrender.com/health`

Should return:
```json
{
  "status": "healthy",
  "model": "loaded",
  "preprocessor": "loaded"
}
```

## If Still Broken
1. Check Render logs - look for "RETRAINING PIPELINE COMPLETED"
2. Make scripts executable:
   ```bash
   chmod +x build.sh start.sh
   git add build.sh start.sh
   git commit -m "Make scripts executable"
   git push
   ```
3. Clear build cache and redeploy

## Files Changed
- `requirements.txt` - Pinned versions
- `build.sh` - Auto-retrain on deploy
- `start.sh` - Production server
- `retrain_models.py` - Retraining logic
- `app.py` - Health check endpoint
- `src/utils.py` - Better error handling

## That's It!
Your app should now work on Render without the AttributeError.

---
For detailed info, see: `DEPLOYMENT.md` and `FIX_SUMMARY.md`
