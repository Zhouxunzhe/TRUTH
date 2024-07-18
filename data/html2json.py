from bs4 import BeautifulSoup
import json

# Read your HTML file
file_name = "gpt4vlmr"

with open(file_name+'.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Parse HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Find all 'trial' divs
trials = soup.find_all('div', class_='trial')

# Initialize a list to store data
data = []

# Process each 'trial' div
for trial in trials:
    prompt = "\n".join([line.strip() for line in trial.find('p', class_='prompt').get_text().split('\n')])
    response = "\n".join([line.strip() for line in trial.find('p', class_='response').get_text().split('\n')])
    if trial.find('p', class_='correct') is not None:
        match = "\n".join([line.strip() for line in trial.find('p', class_='correct').get_text().split('\n')])
    else:
        match = "\n".join([line.strip() for line in trial.find('p', class_='incorrect').get_text().split('\n')])

    # Initialize an entry for this trial
    trial_entry = {
        'Prompt': prompt,
        'Response': response,
        'Match': match
    }

    # Process images
    images = trial.find_all('img')
    for idx, img in enumerate(images):
        trial_entry[f'Image{idx + 1}'] = img['src']

    # Append data to the list
    data.append(trial_entry)

# Create JSON structure
json_data = {'data': data}

# Write JSON to a file
with open(file_name+'.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"Conversion successful. JSON file created: '{file_name}.json'")
