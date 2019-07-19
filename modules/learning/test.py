with open("xres.txt", "r", encoding="utf8") as file:
    content = file.read().replace("\n", "").replace("  ", " ").replace(" ", ",")

with open("xres.py", "w", encoding="utf8") as file:
    print("xk = " + content, file=file)

