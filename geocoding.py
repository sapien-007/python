import requests
import json
import urllib

base_url= "https://maps.googleapis.com/maps/api/geocode/json?"
AUTH_KEY = "AIzaSyD5ivTw6bITbvi2GDQfuPrpcArxSS3BCzw"

# set up your search parameters - address and API key
parameters = {"address": "Tuscany, Italy",
              "key": AUTH_KEY}

# urllib.parse.urlencode turns parameters into url
print(f"{base_url}{urllib.parse.urlencode(parameters)}")


r = requests.get(f"{base_url}{urllib.parse.urlencode(parameters)}")

data = json.loads(r.content)
data

data.get("results")[0].get("geometry").get("location")
