# 📋 Heroku Deployment Checklist for Traffic Analyzer

## 🔧 Pre-Deployment Setup

### 1. Required Files in Root Directory
- [ ] `Procfile` exists with: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- [ ] `runtime.txt` exists with: `python-3.11.9`
- [ ] `requirements.txt` exists with all dependencies
- [ ] `Aptfile` exists with system packages
- [ ] `.gitignore` excludes unnecessary files

### 2. File Content Verification

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.11.9
```

**requirements.txt:**
```
opencv-python-headless
ultralytics
streamlit
yt-dlp
pandas
pillow
numpy
```

**Aptfile:**
```
libgl1-mesa-glx
libglib2.0-0
ffmpeg
libsm6
libxext6
```

**.gitignore:**
```
__pycache__/
*.pyc
.env
.DS_Store
*.log
node_modules/
```

## 🚀 Heroku Account Setup

### 3. Heroku Prerequisites
- [ ] Heroku account created at [heroku.com](https://heroku.com)
- [ ] Heroku CLI installed on your machine
- [ ] Git installed and configured
- [ ] Logged into Heroku CLI: `heroku login`

### 4. Create Heroku App
```bash
# Navigate to your project directory
cd c:\Users\User\Desktop\video_YOLO_v1

# Create new Heroku app
heroku create your-app-name

# Or for automatic name generation
heroku create
```

## 📦 Code Preparation

### 5. App Configuration
- [ ] `app.py` is your main file (not `og_app.py`)
- [ ] Dynamic model loading implemented (no large .pt files)
- [ ] Environment variables configured properly
- [ ] Port configuration for Heroku in app code

### 6. Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = $PORT
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false

[browser]
gatherUsageStats = false
```

## 🔨 Deployment Steps

### 7. Git Repository Setup
```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit for Heroku deployment"
```

### 8. Deploy to Heroku
```bash
# Add Heroku remote
git remote add heroku https://git.heroku.com/your-app-name.git

# Deploy to Heroku
git push heroku main

# Or if your branch is master
git push heroku master
```

### 9. Set Environment Variables (if needed)
```bash
# Example for any API keys or secrets
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
```

## 🔍 Post-Deployment Verification

### 10. Check Deployment Status
```bash
# View logs
heroku logs --tail

# Check app status
heroku ps

# Open app in browser
heroku open
```

### 11. Test App Functionality
- [ ] App loads without errors
- [ ] YOLO model downloads successfully
- [ ] Stream connection works
- [ ] All features functional
- [ ] No memory/timeout issues

## 🐛 Troubleshooting Commands

### 12. Debug Common Issues
```bash
# View recent logs
heroku logs --tail

# Check dyno status
heroku ps

# Restart app
heroku restart

# Check build logs
heroku builds

# Access Heroku bash (for debugging)
heroku run bash
```

### 13. Common Error Solutions

**Slug size too large:**
```bash
# Check what's in your build
heroku builds:info BUILD-ID
```

**App crashed:**
```bash
# Check logs for errors
heroku logs --tail --num=200
```

**Port binding error:**
- [ ] Verify Procfile has correct port configuration
- [ ] Check `app.py` handles `$PORT` environment variable

## 📊 Performance Optimization

### 14. Scaling (Optional)
```bash
# Scale to hobby dyno (free tier)
heroku ps:scale web=1

# Upgrade dyno type if needed (paid)
heroku ps:type Standard-1X
```

### 15. Add-ons (Optional)
```bash
# Add logging (free tier)
heroku addons:create papertrail:choklad

# Add monitoring
heroku addons:create newrelic:wayne
```

## ✅ Final Checklist

### 16. Deployment Success Verification
- [ ] No build errors in logs
- [ ] App accessible via Heroku URL
- [ ] YOLO models load correctly
- [ ] YouTube stream connections work
- [ ] All tracking algorithms functional
- [ ] Spatial analysis works
- [ ] No memory limit exceeded
- [ ] Response times acceptable

### 17. Maintenance Setup
- [ ] Set up automatic deployments from GitHub (optional)
- [ ] Configure monitoring/alerts
- [ ] Document any environment-specific configurations
- [ ] Plan for model updates/maintenance

## 🎯 Quick Deploy Commands Summary

```bash
# Complete deployment in one go
heroku create your-app-name
git add .
git commit -m "Deploy to Heroku"
git push heroku main
heroku open
```

## 🚨 Red Flags to Watch For

- ❌ Slug size > 500MB
- ❌ Build time > 15 minutes
- ❌ Memory usage > 512MB (free tier)
- ❌ Response time > 30 seconds
- ❌ Frequent crashes in logs

---

## 📝 Notes

- **Model Loading**: Your app uses dynamic model loading to keep slug size under 500MB
- **Stream Issues**: YouTube stream connection may vary between local and cloud environments
- **Performance**: First model load takes ~30-60 seconds due to download
- **Debugging**: Use `heroku logs --tail` to monitor real-time issues

## 🔗 Useful Links

- [Heroku Dev Center](https://devcenter.heroku.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [Heroku CLI Reference](https://devcenter.heroku.com/articles/heroku-cli-commands)

---

**Last Updated**: October 2025  
**App Version**: Traffic Analyzer v1.0  
**Deployment Target**: Heroku Free Tier