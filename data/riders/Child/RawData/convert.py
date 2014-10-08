# this simply scales Charlie's input file with respect to the average height of
# a five year old male.
scale = 1.016 / 1.758;

f = open('CharlieYeadonMeas.txt', 'r')
n = open('ChildYeadonMeasScaled.txt', 'w')

for line in f:
    if line.startswith('L'):
        var, val = line.strip().split(':')
        scaled = scale * float(val)
        n.write(var + ': ' + str(scaled) + '\n')
    else:
        n.write(line)

f.close()
n.close()
