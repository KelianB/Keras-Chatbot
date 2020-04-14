import re
from string import punctuation

import config as cfg

NAMES = ["cole", "laurie", "loretta", "cornelius", "brian", "walter", "carl", "sam", "tom", "jeffrey", "fred", "cole", "kevin", "jake", "billy", "kathy", "james", "annie", "otis", "wolfi", "michael", "marry", "johnson", "jerry", "stanzi", "paula", "jeff", "smith", "mary", "rachel", "milo", "claire", "davis", "tommy", "paul", "johnson", "casey", "harrington", "rick", "david", "jeffrey", "jack", "sid", "rose", "mikey", "marty", "dave", "jones", "enzo", "mike", "betty", "bill", "amon", "cosgrove", "bobby", "romeo", "rudy", "elaine", "jeffrey", "jim", "tom", "mickey", "ronnie", "cindy", "paulie", "jimmy", "alex", "ted", "stella", "joe", "ed", "benjamin", "ike", "richard", "gale", "johnny", "walter", "george", "frank", "dignan", "johnny", "norman", "bob", "john", "louis", "bruce", "paulie", "charlie", "charles", "christ", "i\\97", "helen", "dolores", "peter", "fred", "nick", "andy", "eddie"]
    
REPLACEMENT_DICT = {
    "won't": "will not",
    "wouldn't": "would not",
    "let's": "let us",
    "where's": "where is",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "'m": " am",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " had",

    "don't it": "doesn't it",
    "'bout": "about",
    "'til": "until",
    "c'mon": "come on",
    "stayin'": "staying",
    "rollin'": "rolling",
    "can't": "cannot",
    "ain't": "are not",
    "n't": " not",
    
    #"'s": ' is', # breaks possessive
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "it's": "it is",

    "o.k.": "ok",

    ",...": ",",
    "...!": " !",
    "..!": " !",
    ".!": " !",
    "...?": " ?",
    "..?": " ?",
    ".?": " ?",
    "EOS": "",
    "BOS": "",
    "eos": "",
    "bos": "",
    ". . .": "...",
    ". .": " ",
    ".  .": " ",
    "<u>": "",
    "</u>": "",
    "<b>": "",
    "</b>": "",
    "<i>": "",
    "</i>": "",
}
repl_by_space_dict = ["-", "_", " *", " /", "* ", "/ ", "\"", "--"]

def cleanup(text):
    if len(text) == 0:
        return ""

    text = text.lower()

    text = text.replace("’", "'").replace("", "'")
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n", "", text)
    
    # Replace double dots with triple dots
    text = re.sub(r"(?<=([a-z]| ))(\.){2}(?=([a-z]| |$))", "... ", text)

    for _ in range(3):
        for v in REPLACEMENT_DICT:
            text = text.replace(v, REPLACEMENT_DICT[v])
    for v in repl_by_space_dict:
        text = text.replace(v, " ")
    # Change multi spaces to single spaces and strip line
    text = re.sub(" +", " ",  text).strip()

    if len(text) > 1 and text[-1] in [","]:
        text = text[:-1].strip()
    
    while len(text) > 1 and text[0] in punctuation:
        text = text[1:].strip()

    # Fix occurences of ".word" (have to be careful with acronyms)
    
    return text




""" Processes a query before it is given as input to the bot. """
def preprocess_query(text, name):
    text = cleanup(text)

    if len(text) == 0:
        return ""

    # at this point the text is all lower case, striped and word contractions are normalized

    # Normalize names
    text = text.replace(", " + cfg.BOT_NAME, "")
    text = text.replace(cfg.BOT_NAME + " ,", "")

    if text[-1] != "!" and text[-1] != "?" and text[-1] != ".":
        text += "."

    # Handle the case where the text is just punctuation
    non_punctuation = 0
    for character in text:
        if not (character in punctuation):
            non_punctuation += 1
    if non_punctuation == 0:
        text = ""
      
    return text

""" Processes a response from the bot, before it is printed to the user. """
def postprocess_response(text, name):
    text = cleanup(text)

    if len(text) == 0:
        return ""

    # at this point the text is all lower case, striped and word contractions are normalized

    # Normalize names
    for person_name in NAMES:
        text = text.replace(", " + person_name, ", " + name)
        text = text.replace(" " + person_name + " ," , " " + name + " ,")
        text = text.replace("i am " + person_name, "i am " + cfg.BOT_NAME)
        text = text.replace("my name is " + person_name, "my name is " + cfg.BOT_NAME)

    if text[-1] != "!" and text[-1] != "?" and text[-1] != ".":
        text += "."

    # Handle the case where the text is just punctuation
    non_punctuation = 0
    for character in text:
        if not (character in punctuation):
            non_punctuation += 1
    if non_punctuation == 0:
        text = "what?"
      
    return text
