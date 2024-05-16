import yaml
import os

print(os.getcwd())
# Load the authtoken from the file
with open('authtoken.txt', 'r') as file:
    authtoken = file.read().strip()

# Define your configuration with the authtoken
config = {
    'version': "2",
    'authtoken': authtoken,
    'tunnels': {
        'front': {
            'proto': 'http',
            'addr': 3000,
            'domain': 'condor-super-halibut.ngrok-free.app'
        },
        'back': {
            'proto': 'http',
            'addr': 8000
        }
    }
}

# Convert the dictionary to a YAML formatted string
yaml_config = yaml.dump(config)

# Output or save the configuration to a file
with open('ngrok.yml', 'w') as file:
    file.write(yaml_config)