import os

def width():
    width, lines = os.get_terminal_size()
    return width

def divPrint():
    print('*'*width())
    print('*'*width())

def centerPrint(s):
    lines = s.split('\n')
    for line in lines:
        print(f"{' ' * ((width() // 2) - (len(line) // 2))}{line}")