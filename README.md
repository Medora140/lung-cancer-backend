# Lung Cancer Detection Backend (Flask)

## Setup
1. Create a new GitHub Repository for this backend.
2. Push all files in this folder to the repository.

## Deployment on Render
1. Log in to [Render](https://render.com).
2. Click **New** > **Web Service**.
3. Connect your GitHub repository.
4. Set the following:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy**.

## API Endpoints
- `GET /health`: Check if the server and model are running.
- `POST /predict`: Upload an image to get predictions and Grad-CAM heatmap.
