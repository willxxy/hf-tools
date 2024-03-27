
### PRIVATE MODELS ACCESS witgh API KEY
print('Loading API key')
with open('./api_keys.txt', 'r') as file:
    file_contents = file.readlines()
api_key = file_contents[0]

login(token = api_key)
