# commons.py (minimal stub for VITS)
def intersperse(lst, item):
    result = [item] * (2 * len(lst) + 1)
    result[1::2] = lst
    return result