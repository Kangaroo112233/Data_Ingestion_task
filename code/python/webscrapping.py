import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Yellow Pages webpage
url = 'https://www.yellowpages.com/search?search_terms=home+health&geo_location_terms=Carrollton%2C+TXE'

# Send a GET request to the webpage
response = requests.get(url)

# Parse HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all elements containing address information
addresses = []
for element in soup.find_all('div', class_='address-class'):
    address = element.text.strip()  # Extract text and remove leading/trailing spaces
    addresses.append(address)

# Convert the addresses list to a pandas DataFrame
df = pd.DataFrame(addresses, columns=['Address'])

# Export DataFrame to Excel or Word
df.to_excel('addresses.xlsx', index=False)  # Export to Excel
# df.to_csv('addresses.csv', index=False)  # Export to CSV
# df.to_csv('addresses.txt', index=False)  # Export to text file

# Alternatively, export to Word using libraries like python-docx
# (code for Word export omitted for brevity)

print('Addresses extracted and saved successfully.')
