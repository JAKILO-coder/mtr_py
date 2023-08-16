import os
from flaskplotlib import app

def runserver():
    port = int(os.environ.get('PORT', 9000))
    app.run(host='143.89.49.63', port=port, debug=True)

if __name__ == "__main__":
    runserver()
