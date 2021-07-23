import numpy as np

def exponent_parser(s):
    outstr = '$'
    pieces = s.split('e')
    if float(pieces[0]) != 1:
        outstr += pieces[0]
    if int(pieces[1][1:]) != 0:
        if outstr == '$':
            outstr += ' 10^{'
        else:
            outstr += ' \\cdot 10^{'
        if pieces[1][0] == "-":
            outstr += '-'
        outstr += '{}'.format(int(pieces[1][1:]))
        outstr += '}'
    if outstr == '$':
        outstr += '1.0'
    outstr += '$'
    return outstr
    
        

datachunk = np.genfromtxt('scalar_table_data.csv', skip_header=1, delimiter=',', dtype=np.unicode_)

for i in range(datachunk.shape[0]):
    this_str = ''

    #figure out the type
    erf = datachunk[i,5]
    sf  = datachunk[i,6]
    if 'False' in erf and 'False' in sf:
        this_str += 'L & '
    elif 'True' in erf and 'False' in sf:
        this_str += 'D & '
    elif 'True' in erf and 'True' in sf:
        this_str += 'D/SF & '

    #add the P, S, R
    for j in [1, 0, 2]:
        this_str += exponent_parser(datachunk[i,j]) + ' & '
    
    #add resolution
    this_str += datachunk[i,8] + ' & '

    #add tsim
    this_str += '$' + datachunk[i,9] + '$ & '

    #add deltas
    this_str += '('
    for j in [10, 11, 12]:  
        this_str += datachunk[i,j][1:]
        if j != 12:
            this_str += ', '
    this_str += ') & '

    #add f
    this_str += datachunk[i,13] + ' & '
    
    #add xi
    this_str += datachunk[i,14] + ' & '

    #add vel
    this_str += datachunk[i,-1] + ' \\\\ '

    print(this_str)

