# Deployment checklist

1. Extract this ZIP so `app.py`, `core.py`, `requirements.txt`, `.python-version`, and `runtime.txt` are at the repository root (not nested inside another directory).
2. Push the files to your GitHub repository's `main` branch.
3. In Streamlit Community Cloud, create/redeploy the app using `app.py` as the main file and select Python 3.12 in Advanced settings.
4. The runtime selection must be made during app creation; if an existing app is locked to another Python version, create a new deployment after saving any secrets/settings.
5. Check logs for successful dependency installation and verify the app loads. Confirm the sensitivity chart shows five points for K=2,5,10,20,25.

The sensitivity chart now explicitly binds the x-axis to `Max Open (K)` and y-axis to `Profit (Rs.)`; this avoids relying on the dataframe index being interpreted as chart data.
