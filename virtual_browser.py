from selenium import webdriver
import signal
import sys
import time

# Define a signal handler to close the browser on Ctrl+C
def signal_handler(sig, frame):
    print('Received Ctrl+C, closing browser...')
    driver.quit()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Launch headless Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.set_capability("goog:loggingPrefs", {  # old: loggingPrefs
    "browser": "INFO"})
driver = webdriver.Chrome(options=options)

driver.set_window_size(1920, 1080)
driver.execute_cdp_cmd("Network.setCacheDisabled", {"cacheDisabled":True})

# Open a local HTML file
driver.get('http://130.126.139.208:8000/view_synthetic.html')  # Replace with your file path

# Optionally, take a screenshot to visually verify
driver.save_screenshot('screenshot.png')
print('Virtual browser is running...')

# Do not close the browser here, so it stays open in the background
while True:
   # Get the browser console logs
    logs = driver.get_log('browser')

    # Print the logs
    for log in logs:
        print(log)

    # Pause for a short duration to avoid constantly polling
    time.sleep(1)