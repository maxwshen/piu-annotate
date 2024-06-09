import pickle


def is_js_lists(inp: any) -> bool:
    """ Returns True if inp is a list containing two lists with same size. """
    is_list = lambda x: type(x) == list
    if is_list(inp):
        if len(inp) == 2:
            if is_list(inp[0]) and is_list(inp[1]):
                if len(inp[0]) == len(inp[1]):
                    return True
    return False


def js_lists_to_dict(ll: list[list]) -> dict:
    keys, values = ll[0], ll[1]
    d = dict()
    for k, v in zip(keys, values):
        if is_js_lists(v):
            d[k] = js_lists_to_dict(v)
        elif type(v) is list and all(is_js_lists(x) for x in v):
            d[k] = [js_lists_to_dict(x) for x in v]
        else:
            d[k] = v
    return d


class PiuCenterStruct:
    def __init__(self, pkl_file: str):
        """ Loads piucenter data struct from pickle file. """
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        self.info_dict = js_lists_to_dict(self.data[0])
        self.chart_card_dict = js_lists_to_dict(self.data[1])
        self.chart_details_lod = [js_lists_to_dict(x) for x in self.data[2]]

    @classmethod
    def from_s3(filename: str):
        """ Alternate constructor: download from s3"""
        return    

    def get_n_sections(self):
        return len(self.chart_details_struct) - 1
