from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# Make browser headless
options = webdriver.ChromeOptions()
options.add_argument('--headless')

driver_path = ChromeDriverManager().install()
driver = webdriver.Chrome(service=webdriver.chrome.service.Service(driver_path), options=options)

# Need to get rss feed from the webpage
# A function to get title, link, 
link = "https://www.thehindu.com/"
criterion = 'div.element'

driver.get(link)
# Wait till the element is present
try:
	elements = WebDriverWait(driver, 10).until(
		EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.element"))
	)
	for i in elements:
		print(i.text)
		print(i.find_element(By.TAG_NAME, "a").get_attribute("href"))
		print()
except Exception as e:
	print(e)

