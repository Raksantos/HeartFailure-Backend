from main import app
import os

if __name__ == "__main__":
  port = os.enviroment.get("PORT", 5000)
  app.run(debug=False, host="0.0.0.0", port=port)