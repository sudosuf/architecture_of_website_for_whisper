
def replaceEngWord(text: str) -> str:
    split_text = text.split(" ")
    print(split_text)
    for word in split_text:
        print(word)
        if word == 'NVN' or 'NVN,' or 'NVN.' :
            text = text.replace("NVN", "НВН")
        if word == 'nvn' or 'nvn,' or 'nvn.':
            text = text.replace("nvn", "НВН")
        if word == 'LZS' or 'LZS,' or 'LZS.':
            text = text.replace("LZS", "ЛЗС")
        if word == "lzs" or "lzs," or "lzs.":
            text = text.replace("lzs", "ЛЗС")
        if word == "zs" or "zs," or "zs.":
            text = text.replace("zs", "ЗС")
        if word == "ZS" or "ZS," or "ZS.":
            text = text.replace("ZS", "ЗС")
        if word == "WNP" or "WNP," or "WNP.":
            text = text.replace("WNP", "ВНП")
        if word == "wnp" or "wnp," or "wnp.":
            text = text.replace("wnp", "ВНП")
    return text
