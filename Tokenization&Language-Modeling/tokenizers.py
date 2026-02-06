import re

# Task 1.1
def corpus_cleaning(text):

    text = re.sub(r'’','\'',text) # smart quotes -> standard quotes
    text = re.sub(r'—','-',text)  # em dash -> standard dash
    text = re.sub(r'<br\s*/?>','',text) # to remove html tags
    text = re.sub(r'&amp;','&',text) # normalizing &amp
    # text = re.sub(r'@[A-Za-z0-9_]+','',text) #to remove x handle usernames
    text = re.sub(r'\s+',' ',text) # removes excessive spacing
    text = text.strip() # removes leading and trailing spaces

    return text



