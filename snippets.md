# wait for element to come
elements = WebDriverWait(driver, 10).until(
		EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.element"))
	)