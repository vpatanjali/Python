from bitarray import bitarray

# Using natural indexing, not computer indexing!

SIZE = 2**32

outfile = open('G:/Kaggle/primes','w')

arr = bitarray(SIZE)

arr.setall(True)
arr[0] = False
arr[1] = False

for i in xrange(SIZE):
	if arr[i]:
		outfile.write(str(i) + '\n')
		outfile.flush()
		arr[2*i:SIZE:i] = False

outfile.close()