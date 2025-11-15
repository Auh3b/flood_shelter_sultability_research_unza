"""
One-at-a-time Modeller of weighted values 
"""
import numpy as np


class OAT:
    def __init__(self, weight_dict):
        self.input = weight_dict
        self.values = None
        self.ids = None
        self._load_weight_dict()

    def _load_weight_dict(self):
        if (self.input == None):
            raise 'Initiate with a weight dictionary'
        self.values = list(self.input.values())
        self.ids = list(self.input.keys())

        return self

    def _per_dec(self, min, max, interval):
        per_range = list(range(min, max, interval))
        per_range.pop(0)
        dec_arr = [x/100 for x in per_range]
        print(dec_arr)
        return dec_arr

    def solve(self, max, min, interval):
        ids = self.ids
        per_range = self._per_dec(min, max, interval)
        _output = []
        for x in ids:
            _sub = []
            for y in per_range:
                _sub.append(self.adjust(x, y))
            _output.append(_sub)
        output = np.array(_output)
        return output

    def adjust(self, crit, value):
        if (self.values == None):
            raise 'Weights need to be added'
        if (crit == None or type(crit) != str):
            raise 'Criterion is not specified or is not a string'
        if (value == None or type(value) != float):
            raise 'Value is not a number'

        #  get id index
        if (crit not in self.ids):
            raise f"{crit} not part of weight dictionary. Please check spelling"

        idx = self.ids.index(crit)
        return self.getAdjustMatrix(idx, value)

    def getAdjustMatrix(self, tdx, pc):
        if (tdx == None):
            raise 'Target index has not been provided.'

        mcc_value = self.values[tdx]
        return [self.getMCCValue(mcc_value, pc) if idx == tdx else self.getOCCValue(mcc_value, x, pc) for idx, x in enumerate(self.values)]

    def getMCCValue(self, x, pc):
        """
        Get the main changing criterion value
        x = main criterion value
        pc = percentage change value
        """
        return x + (x*pc)

    def getOCCValue(self, x, v, pc):
        """
        Get the other changing criterion
        x = main criterion value
        v = 
        pc = percentage change value
        """
        return (1 - self.getMCCValue(x, pc)) * v/(1 - x)
