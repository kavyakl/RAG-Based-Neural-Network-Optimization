import requests

API_KEY = 'ei_98e3a57d572b6b95f1b72b13b1124d50acf2ed6660896076934f7e35fc4fc208'
PROJECT_ID = 669207  # Correct project ID

headers = {
    "x-api-key": API_KEY
}

url = f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("✅ API ping successful!")
    print("Project Info:", response.json())
else:
    print(f"❌ API call failed with status code {response.status_code}")
    print(response.text)