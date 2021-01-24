#Based on code from Stack Overflow: https://stackoverflow.com/questions/6512280/accept-a-range-of-numbers-in-the-form-of-0-5-using-pythons-argparse
#Implement a numeric range parser for argparse, allowing a range of numbers to
#be included

import argparse
import re

def parseNumList(string):
    #Look for two numeric elements on either side of a dash:
        # (\d+): match one or more digits, save the result (match \1 or $1)
        #(?:...) match whatever else is in the parentheses without
        #creating a new match object (i.e. this avoids e.g. -3 becoming
        #reference \2 or $2
        #(?-(\d+))? matches 0 or 1 occurence of a dash followed by some number
        #of digits. Saves the digits, not the dash.
        #$ matches end of string
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m: 
        raise argparse.ArgumentTypeError("'" + string + "' is not a range \
                                         of numbers.\
                                Expected forms like '0-5' or '2'.")
    #The first digit match is the start of the range
    start = m.group(1) #group(0) is the whole match
    end = m.group(2) or start #if m.group(2) is None, end = start
    #Specify that the ints are base 10
    return list(range(int(start, 10), int(end, 10) + 1))

#Rather than add objects to a list, concatenate objects onto a list
class concat(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        new_values = getattr(namespace, self.dest, [])
        if new_values is None:
            new_values = []
        #Flatten a list of lists to a single list
        for sublist in values:
            for item in sublist:
                new_values.append(item)
        setattr(namespace, self.dest, new_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--indexes',type=parseNumList,action=concat,
                        nargs='+',
                        help="pass a range of numbers e.g. '0 2 5-7' would \
                        yield [0,2,5,6,7]")
    parser.add_argument('-p','--temp')
    args = parser.parse_args()
    print(args)



