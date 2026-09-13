import urllib.request

print("Starting download of the official text matrix...")
print("This file is large and will take 2 to 3 minutes. Please wait.")

# The full link is safely contained inside the script where it won't get cut off!
url = "https://githubusercontent.com"

urllib.request.urlretrieve(url, "glove.6B.200d.txt")

print("REAL DATA DOWNLOAD COMPLETE!")
