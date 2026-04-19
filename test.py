import requests
import re

def scrape(username):
    url = f"https://www.instagram.com/{username}/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    r = requests.get(url, headers=headers)
    
    desc = re.search(r'content="([^"]*Followers,[^"]*Following,[^"]*Posts[^"]*)"', r.text)
    if desc:
        print("Meta og:description:", desc.group(1).encode('utf-8', errors='ignore'))
    else:
        print("Meta og:description NOT found")

scrape('nishaant_mishra_')
