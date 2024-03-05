# import pandas as pd
# import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common import keys
from bs4 import BeautifulSoup
import time
import re


class sensortower:
    def __init__(self, url, driver):
        self.url = url
        self.driver = driver

        self.country_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[3]/div/div/div/div/div[2]/button'
        self.country_button_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[3]/div/div/div/div/div[2]/button'
        self.country_input_XPath = '//*[@id=":rf:"]'
        self.country_listbox_XPath = '//*[@id=":rf:-listbox"]'  # headless
        self.category_XPath = (
            '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[4]/div/div'
        )
        self.category_button_XPath = (
            '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[4]/div/div'
        )
        self.category_input_XPath = ""
        self.subcat1_XPath = '//*[@id="menu-"]/div[3]/ul/li[13]'  # headless
        self.subcat2_XPath = '//*[@id="menu-"]/div[3]/ul/li[35]'  # headless

        self.cookies_XPath = "osano-cm-accept-all"
        self.table_XPath = '//*[@id="mainContent"]/div[1]/div/div[4]/div/div[1]'

    def get_country(self):
        # Open url
        # self.driver.get(self.url)
        try:
            # Find the button element for the dropdown toggle
            dropdown_countries = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, self.category_button_XPath))
            )
            dropdown_countries = self.driver.find_element(
                By.XPATH, self.country_button_XPath
            )
            dropdown_countries.click()

            # countries_ul = driver.find_element(By.XPATH, '//*[@id=":r7:-listbox"]')
            countries_ul = self.driver.find_element(
                By.XPATH, self.country_listbox_XPath
            )  # headless

            # Locate all the <li> elements within the <ul> element
            country_elements = countries_ul.find_elements(By.TAG_NAME, "li")

            # Extract country names and populate them into a list
            country_names = [country.text for country in country_elements]

            # Print the list of country names (for verification)
            # print("country names:", country_names)

            time.sleep(0.1)
            dropdown_countries = self.driver.find_element(
                By.XPATH, self.country_button_XPath
            )
            dropdown_countries.click()

            return country_names

        except Exception as e:
            print(f"Error: {e}")

    def get_category(self):
        # Open url
        # self.driver.get(self.url)

        # Due to cookies blocking accepting terms
        cookies_accept_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, self.cookies_XPath))
        )
        cookies_accept_button.click()

        # Find the button element for the dropdown toggle
        dropdown_categories = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, self.category_button_XPath))
        )
        dropdown_categories.click()

        # Locate the categories dropdown menu
        categories_menu = self.driver.find_element(
            By.XPATH, '//*[@id="menu-"]/div[3]/ul'
        )

        # Locate the subcategory dropdown menu and scroll into view
        subcategories_menu1 = self.driver.find_element(
            By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[13]'
        )  # headless
        # subcategories_menu1 = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH,  '//*[@id="menu-"]/div[3]/ul/li[13]')))
        self.driver.execute_script(
            "arguments[0].scrollIntoView();", subcategories_menu1
        )
        subcategories_menu1.click()

        # Locate the 2nd subcategory dropdown menu and scroll into view
        # subcategories_menu2 = self.driver.find_element(By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[35]')
        subcategories_menu2 = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[35]'))
        )  # headless
        self.driver.execute_script(
            "arguments[0].scrollIntoView();", subcategories_menu2
        )
        subcategories_menu2.click()

        category_elements = categories_menu.find_elements(By.TAG_NAME, "li")

        # Print the list of categories (for verification)
        category_list = [
            category.text for category in category_elements if category.text.strip()
        ]
        # print("categories:", category_list)

        return category_elements

    def scrape_page(self):
        # Open url
        # self.driver.get(self.url)

        # Use WebDriverWait to ensure the table elements are visible
        table = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_all_elements_located((By.XPATH, self.table_XPath))
        )

        # dropdown = self.driver.find_element(By.XPATH, self.country_XPath)
        # print(dropdown)

        # Initialize empty list and iterate through table elements
        app_ids = []
        for row in table:
            # Finding <a> class attributes
            links = row.find_elements(By.TAG_NAME, "a")
            for link in links:
                # Extract href links>
                href = link.get_attribute("href")
                href = re.split(r"[/?]", href)[4]
                app_ids.append(href)

        return app_ids

    def process_data(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(100)

        countries = self.get_country()
        for country in countries:
            print(country)

        # Shutdown chromium driver
        self.driver.quit()


if __name__ == "__main__":
    # Initialize webdriver running headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_service = Service()
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    # driver = webdriver.Chrome(service=chrome_service)
    # driver = webdriver.Chrome()
    # driver.maximize_window()
    url = "https://app.sensortower.com/top-charts?country=US&category=0&date=2024-03-05&device=iphone&os=android"

    scraper = sensortower(url, driver)
    scraper.process_data()
    # print(scraper.scrape_page())
