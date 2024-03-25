import pandas as pd

# import numpy as np
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import random
import time
import re


countries = {
    "US": "US",
    "Australia": "AU",
    "Canada": "CA",
    "China": "CN",
    "France": "FR",
    "Germany": "DE",
    "United Kingdom": "UK",
    "Italy": "IT",
    "Japan": "JP",
    "Russia": "RU",
    "South Korea": "KR",
    "Afghanistan": "AF",
    "Algeria": "DZ",
    "Angola": "AO",
    "Argentina": "AQ",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Bahrain": "BH",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Bermuda": "BM",
    "Bolivia": "BO",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Cambodia": "KH",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Estonia": "EE",
    "Finland": "FI",
    "Georgia": "GE",
    "Ghana": "GH",
    "Greece": "GR",
    "Guatemala": "GT",
    "Hong Kong": "HK",
    "Hungary": "HU",
    "India": "IN",
    "Indonesia": "ID",
    "Iraq": "IQ",
    "Ireland": "IE",
    "Israel": "IL",
    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kuwait": "KW",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Libya": "LY",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Macau": "MO",
    "Madagascar": "MG",
    "Malaysia": "MY",
    "Malta": "MT",
    "Mexico": "MX",
    "Morocco": "MA",
    "Mozambique": "MZ",
    "Myanmar": "MM",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Nigeria": "NG",
    "Norway": "NO",
    "Oman": "OM",
    "Pakistan": "PK",
    "Panama": "PA",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Romania": "RO",
    "Saudi Arabia": "SA",
    "Serbia": "RS",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "South Africa": "ZA",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Thailand": "TH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
    "Venezuela": "VE",
    "Vietnam": "VN",
    "Yemen": "YE",
}


categories_google = {
    "Overall": "all",
    "Art & Design": "art_and_design",
    "Auto & Vehicles": "auto_and_vehicles",
    "Beauty": "beauty",
    "Books & Reference": "books_and_reference",
    "Business": "business",
    "Comics": "comics",
    "Communication": "communication",
    "Dating": "dating",
    "Education": "education",
    "Entertainment": "entertainment",
    "Events": "events",
    "All Family": "family",
    "Family / Action & Adventure": "family_action",
    "Family / Brain Games": "family_braingames",
    "Family / Creativity": "family_create",
    "Family / Education": "family_education",
    "Family / Music & Video": "family_musicvideo",
    "Family / Pretend Play": "family_pretend",
    "Family / Ages 5 & Under": "family_age_range1",
    "Kids / Ages 6-8": "family_age_range2",
    "Kids / Ages 9-11": "family_age_range3",
    "Finance": "finance",
    "Food & Drink": "food_and_drink",
    "All Games": "game",
    "Game / Action": "game_action",
    "Game / Adventure": "game_adventure",
    "Game / Arcade": "game_arcade",
    "Game / Board": "game_board",
    "Game / Card": "game_card",
    "Game / Casino": "game_casino",
    "Game / Casual": "game_casual",
    "Game / Educational": "game_educational",
    "Game / Music": "game_music",
    "Game / Puzzle": "game_puzzle",
    "Game / Racing": "game_racing",
    "Game / Role Playing": "game_role_playing",
    "Game / Simulation": "game_simulation",
    "Game / Sports": "game_sports",
    "Game / Strategy": "game_strategy",
    "Game / Trivia": "game_trivia",
    "Game / Word": "game_word",
    "Health & Fitness": "health_and_fitness",
    "House & Home": "house_and_home",
    "Libraries & Demo": "libraries_and_demo",
    "Lifestyle": "lifestyle",
    "Maps & Navigation": "maps_and_navigation",
    "Medical": "medical",
    "Music & Audio": "music_and_audio",
    "News & Magazines": "news_and_magazines",
    "Parenting": "parenting",
    "Personalization": "personalization",
    "Photography": "photography",
    "Productivity": "productivity",
    "Shopping": "shopping",
    "Social": "social",
    "Sports": "sports",
    "Tools": "tools",
    "Travel & Local": "travel_and_local",
    "Weather": "weather",
    "Video Players & Editors": "video_players",
}


categories_apple = {
    "All Categories": "0",
    "Books": "6018",
    "Business": "6000",
    "Developer Tools": "6026",
    "Education": "6017",
    "Entertainment": "6016",
    "Finance": "6015",
    "Food & Drink": "6023",
    "All Games": "6014",
    "Games / Action": "7001",
    "Games / Adventure": "7002",
    "Games / Board": "7004",
    "Games / Card": "7005",
    "Games / Casino": "7006",
    "Games / Casual": "7003",
    "Games / Family": "7009",
    "Games / Music": "7011",
    "Games / Puzzle": "7012",
    "Games / Racing": "7013",
    "Games / Role Playing": "7014",
    "Games / Simulation": "7015",
    "Games / Sports": "7016",
    "Games / Strategy": "7017",
    "Games / Trivia": "7018",
    "Games / Word": "7019",
    "Graphics & Design": "6027",
    "Health & Fitness": "6013",
    "All Ages": "9007",
    "Kids / Ages 5 & Under": "10000",
    "Kids / Ages 6-8": "10001",
    "Kids / Ages 9-11": "10002",
    "Lifestyle": "6012",
    "Medical": "6020",
    "Music": "6011",
    "Navigation": "6010",
    "News": "6009",
    "Photo & Video": "6008",
    "Productivity": "6007",
    "Reference": "6006",
    "Social Networking": "6005",
    "Shopping": "6024",
    "Sports": "6004",
    "Travel": "6003",
    "Utilities": "6002",
    "Weather": "6001",
}


class sensortower:
    def __init__(self, url, driver):
        self.url = url
        self.driver = driver

        self.country_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[3]/div/div/div/div/div[2]/button'
        self.country_button_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[3]/div/div/div/div/div[2]/button'
        self.country_input_XPath = '//*[@id=":rf:"]'
        # self.country_listbox_XPath = '//*[@id=":rf:-listbox"]'
        self.country_listbox_XPath = '//*[@id=":r7:-listbox"]'  # headless
        # self.country_listbox_XPath = '//*[@id=":r7:"]'

        self.category_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[4]/div/div'
        self.category_button_XPath = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[4]/div/div'
        self.category_input_XPath = ""
        self.subcat1_XPath = '//*[@id="menu-"]/div[3]/ul/li[13]'  # headless
        self.subcat2_XPath = '//*[@id="menu-"]/div[3]/ul/li[35]'  # headless

        self.google_button = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[1]/div/button[2]'
        self.apple_button = '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[1]/div/button[1]'

        self.cookies_XPath = "osano-cm-accept-all"
        self.table_XPath = '//*[@id="mainContent"]/div[1]/div/div[4]/div/div[1]'

        self.google_store_url = "https://app.sensortower.com/top-charts?category={}&country={}&os=android"

    def get_country(self):
        # Open url
        # self.driver.get(self.url)
        try:
            # Find the button element for the dropdown toggle
            # dropdown_countries = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, self.country_button_XPath)))
            # dropdown_countries = self.driver.find_element(By.XPATH, self.country_button_XPath)
            # time.sleep(2.5)
            # dropdown_countries = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, self.country_button_XPath)))
            # dropdown_countries.click()

            # time.sleep(5)

            print("Retrieve ul elements")
            # dropdown_countries = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, self.country_button_XPath)))
            # dropdown_countries.click()

            time.sleep(5)

            input_field = driver.find_element(
                By.XPATH, '//*[@id="mainContent"]/div[1]/div/div[2]/div/div[3]/div/div/div/div/div[2]/button'
            )
            # input = input_field.find_element(By.XPATH, '//*[@id=":r3:"]')
            # input_field = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, ":r3:")))
            # input_field = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[4]/div/div/div[1]/div/div[1]/div/div[2]/div/div[3]/div/div/div/div/input')))

            time.sleep(2)
            input_field.click()
            print(input_field.text)
            print("found input field")
            # time.sleep(1)
            time.sleep(1)
            time.sleep(2)
            # countries_ul = driver.find_element(By.XPATH, '//*[@id=":r7:-listbox"]') # headless
            # countries_ul = WebDriverWait(self.driver,30).until(EC.presence_of_element_located((By.ID, ':r3:-listbox')))
            countries_ul = WebDriverWait(self.driver, 60).until(
                EC.presence_of_element_located((By.ID, ":r7:-listbox"))
            )
            # countries_ul = self.driver.find_element(By.XPATH, self.country_listbox_XPath)
            # countries_ul = self.driver.find_element(By.XPATH, '//*[@id=":rf:-listbox"]')
            # countries_ul = self.driver.find_element(By.XPATH, '//*[@id=":r3:-listbox"]')
            # countries_ul = self.driver.find_element(By.XPATH, '//*[@id=":r7:-listbox"]')
            print("Found listbox")
            print("Retrieved ul elements")

            # Locate all the <li> elements within the <ul> element
            country_elements = countries_ul.find_elements(By.TAG_NAME, "li")
            print("Located all elements")

            # Extract country names and populate them into a list
            country_names = [country.text for country in country_elements]

            # Print the list of country names (for verification)
            # print("country names:", country_names)

            time.sleep(0.1)
            dropdown_countries = self.driver.find_element(By.XPATH, self.country_button_XPath)
            dropdown_countries.click()

            return country_names

        except Exception as e:
            print(f"Error: {e}")

    def get_category(self):
        # Open url
        # self.driver.get(self.url)

        # Due to cookies blocking, accepting terms
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
        categories_menu = self.driver.find_element(By.XPATH, '//*[@id="menu-"]/div[3]/ul')

        # Locate the subcategory dropdown menu and scroll into view
        subcategories_menu1 = self.driver.find_element(By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[13]')  # headless
        # subcategories_menu1 = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH,  '//*[@id="menu-"]/div[3]/ul/li[13]')))
        self.driver.execute_script("arguments[0].scrollIntoView();", subcategories_menu1)
        subcategories_menu1.click()

        # Locate the 2nd subcategory dropdown menu and scroll into view
        # subcategories_menu2 = self.driver.find_element(By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[35]')
        subcategories_menu2 = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="menu-"]/div[3]/ul/li[35]'))
        )  # headless
        self.driver.execute_script("arguments[0].scrollIntoView();", subcategories_menu2)
        subcategories_menu2.click()

        category_elements = categories_menu.find_elements(By.TAG_NAME, "li")

        # Print the list of categories (for verification)
        category_list = [category.text for category in category_elements if category.text.strip()]
        # print("categories:", category_list)

        return category_elements

    def scrape_page(self, country, store, button, category, os):
        # Open url
        # self.driver.get(self.url)

        # Use WebDriverWait to ensure the table elements are visible
        # table = WebDriverWait(self.driver, 30).until(
        #     EC.visibility_of_all_elements_located((By.XPATH, self.table_XPath))
        # )

        # table = WebDriverWait(self.driver, 30).until( # ORIGINAL
        #     EC.visibility_of_all_elements_located(
        #         (By.XPATH, '//*[@id="mainContent"]/div[1]/div/div[3]/div/div[1]/table')
        #     )
        # )

        # Wait for page to load
        print("Sleep:", t := random.randint(3, 5), "s")
        time.sleep(t)

        # Open country and category
        self.driver.get(f"https://app.sensortower.com/top-charts?category={category}&country={country}&os={os}")
        time.sleep(5)

        # Wait for button to be clickable
        elem = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, button)))
        elem.click()
        time.sleep(5)

        # elem.click()
        elem.send_keys(Keys.END)
        time.sleep(t)
        elem.send_keys(Keys.END)
        time.sleep(t)
        elem.send_keys(Keys.END)
        time.sleep(t)
        elem.send_keys(Keys.END)
        time.sleep(t)
        elem.send_keys(Keys.HOME)
        elem.send_keys(Keys.HOME)
        elem.send_keys(Keys.HOME)
        elem.send_keys(Keys.HOME)
        time.sleep(2)

        # Use for minimized window
        table = WebDriverWait(self.driver, 30).until(
            EC.visibility_of_all_elements_located(
                (By.XPATH, '//*[@id="mainContent"]/div[1]/div/div[4]/div/div[1]/table/tbody')
            )
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
        app_ids.pop(0)
        apps = pd.DataFrame(app_ids)
        # apps.drop(index=1)
        print("APPS:\n", apps)

        apps.to_csv(f"scrapes/{country}_{store}_{category}.csv", index=False)

        return app_ids

    def process_data(self):
        # self.driver.get(self.url)
        # self.driver.implicitly_wait(100)

        # countries = self.get_country()
        # print(countries)
        url = "https://app.sensortower.com/top-charts?country="
        signin = "https://app.sensortower.com/users/sign_in"
        # signin = "https://app.sensortower.com/users/auth/google_oauth2"
        google_inputfield = '//*[@id="identifierId"]'
        next_button = '//*[@id="identifierNext"]/div/button'
        pwd_input = '//*[@id="password"]/div[1]/div/div[1]/input'
        pwd_next = '//*[@id="passwordNext"]/div/button'

        self.driver.get(signin)
        time.sleep(3)
        google_input = WebDriverWait(self.driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[2]/footer/div[2]/div[1]/a"))
        )
        time.sleep(3)
        google_input.click()
        time.sleep(3)
        google_input = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, google_inputfield)))
        time.sleep(2)
        google_input.click()
        time.sleep(2)
        google_input.send_keys("07dondraper07@gmail.com")
        time.sleep(2)
        google_input = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, next_button)))
        time.sleep(2)
        google_input.click()
        time.sleep(2)
        google_input = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, pwd_input)))
        time.sleep(2)
        google_input.click()
        time.sleep(2)
        google_input.send_keys("StorMand8")
        time.sleep(2)
        google_input = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, pwd_next)))
        time.sleep(2)
        google_input.click()
        time.sleep(5)

        try:
            container = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "MuiDialog-container")))

            # Find the buttons within the container
            buttons = container.find_elements(By.TAG_NAME, "button")

            # Iterate through the buttons and click the one with the desired text or class
            for button in buttons:
                if "Okay" in button.text:
                    button.click()
                    break

            print("Found the button!")
        except Exception as e:
            print(f"Error occured: {e}", "popup not accessible")

        time.sleep(5)
        notification = WebDriverWait(self.driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[5]/div/div/button"))
        )
        time.sleep(2)
        notification.click()
        # Due to cookies blocking, accepting terms
        cookies_accept_button = WebDriverWait(self.driver, 30).until(
            EC.element_to_be_clickable((By.CLASS_NAME, self.cookies_XPath))
        )
        cookies_accept_button.click()
        time.sleep(5)
        self.driver.get(self.url)
        time.sleep(1)

        for country in countries:
            print("next country:", country, countries[country])
            time.sleep(5)
            try:
                self.driver.get(url + countries[country])
                time.sleep(5)

                for category in categories_apple:
                    time.sleep(5)
                    self.scrape_page(countries[country], "apple", self.apple_button, categories_apple[category], "ios")
                print("Successfully scraped apple store")

                time.sleep(5)

                for category in categories_google:
                    time.sleep(5)
                    self.scrape_page(
                        countries[country], "google", self.google_button, categories_google[category], "android"
                    )
                print("Successfully scraped google store")

            except Exception as e:
                print("Error:", e, "\nPossibly no top chart for this country")

        # Shutdown chromium driver
        self.driver.quit()


if __name__ == "__main__":
    # Initialize webdriver running headless
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("window-size=960x1080")
    # chrome_service = Service()
    # driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    # driver = webdriver.Chrome(service=chrome_service)
    driver = uc.Chrome(headless=False, use_subprocess=False)
    # driver.maximize_window()
    driver.set_window_size(1080, 1920)
    url = "https://app.sensortower.com/top-charts?country=US&category=0&date=2024-03-11&device=iphone&os=android"

    scraper = sensortower(url, driver)
    scraper.process_data()
    # print(scraper.scrape_page())
