import sys
fn, sub, tv = sys.argv[1], sys.argv[2], sys.argv[3]
block = '''    if (__dacpp_mpi_is_root_rank()) {
        std::vector<double> __chk; __TENSOR__.tensor2Array(__chk);
        unsigned long long __h = 1469598103934665603ULL;
        double __s=0,__mn=1e300,__mx=-1e300;
        for (size_t __i=0;__i<__chk.size();++__i){ double __v=__chk[__i]; __s+=__v; if(__v<__mn)__mn=__v; if(__v>__mx)__mx=__v; unsigned long long __b; std::memcpy(&__b,&__v,8); __h^=__b; __h*=1099511628211ULL; }
        std::printf("[CHK] n=%zu fnv=%016llx sum=%.17g min=%.17g max=%.17g\\n", __chk.size(), __h, __s, __mn, __mx);
        std::fflush(stdout);
    }'''.replace('__TENSOR__', tv)
# read/write as latin-1 so non-UTF-8 (GBK) comment bytes in the sources round-trip intact
data = open(fn, 'rb').read().decode('latin-1')
lines = data.split('\n')
out = []
inserted = False
have_cstring = ('#include <cstring>' in data)
for l in lines:
    out.append(l)
    if (not have_cstring) and l.strip() == '#include <cstdio>':
        out.append('#include <cstring>')
        have_cstring = True
    if (not inserted) and (sub in l):
        out.append(block)
        inserted = True
open(fn, 'wb').write('\n'.join(out).encode('latin-1'))
sys.stderr.write('INJECT %s: inserted=%s\n' % (fn, inserted))
if not inserted:
    sys.exit(3)
