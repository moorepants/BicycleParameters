import bicycleparameters.bicycleparameters as bp
import timeit
import profile

stratos = bp.Bicycle('Stratos', pathToData='../data')

par = stratos.parameters['Benchmark']

s = "M, C1, K0, K2 = bp.benchmark_par_to_canonical(par)"
profile.run(s)
t = timeit.Timer(s, "from __main__ import bp.benchmark_par_to_canonical")
print "%.2f usec/pass" % (1000000 * t.timeit(number=100000)/100000)
