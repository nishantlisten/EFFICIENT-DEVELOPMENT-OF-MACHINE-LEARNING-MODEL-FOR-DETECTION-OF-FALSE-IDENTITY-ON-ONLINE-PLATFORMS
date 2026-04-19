import requests
import re

def parse_instagram_html(username):
    url = f"https://www.instagram.com/{username}/"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'})
    
    desc_match = re.search(r'<meta property="og:description" content="(.*?)"', r.text)
    if desc_match:
        content = desc_match.group(1)
        print("Found:", content)
        
        # parse: "1,234 Followers, 45 Following, 67 Posts - See Instagram photos and videos from Full Name (@username)"
        stats_match = re.match(r'^([\d.,\w]+)\s+Followers,\s+([\d.,\w]+)\s+Following,\s+([\d.,\w]+)\s+Posts', content)
        if stats_match:
            print("Followers:", stats_match.group(1))
            print("Following:", stats_match.group(2))
            print("Posts:", stats_match.group(3))
    else:
        print("No og:description found")

parse_instagram_html('nishaant_mishra_')
